from app.agents.classifier import ClassifierAgent
from app.agents.dispatch import DispatchAgent
from app.agents.reject_reason import RejectReasonAgent
from app.agents.reply import ReplyAgent
from app.agents.retrieval import RetrievalAgent
from app.core.enums import DecisionSource
from app.core.enums import AcceptSuggestion, ReasonType
from app.core.schemas import ClassificationResult, ComplaintAnalyzeRequest


def test_classifier_accepts_market_complaint() -> None:
    request = ComplaintAnalyzeRequest(problem_text="在超市购买食品后发现过期，要求退款赔偿。")
    result, review_reasons = ClassifierAgent().classify(request)

    assert result.is_market is True
    assert result.accept_suggestion in {AcceptSuggestion.ACCEPT, AcceptSuggestion.REVIEW}
    assert result.confidence >= 0.5
    assert result.decision_source.value in {"MODEL", "RULE"}


def test_classifier_rejects_out_of_scope() -> None:
    request = ComplaintAnalyzeRequest(problem_text="小区物业收取停车费不合理，要求住建部门处理。")
    result, review_reasons = ClassifierAgent().classify(request)

    assert result.is_market is True
    assert result.accept_suggestion == AcceptSuggestion.REVIEW
    assert result.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    assert "命中第16条可能不予受理规则，需人工确认" in review_reasons


def test_classifier_marks_already_processed_for_review() -> None:
    request = ComplaintAnalyzeRequest(problem_text="同一事项法院已受理，消费者再次投诉要求市场监管处理。")
    result, review_reasons = ClassifierAgent().classify(request)

    assert result.accept_suggestion == AcceptSuggestion.REVIEW
    assert result.reason_type == ReasonType.ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED
    assert review_reasons


def test_classifier_marks_not_consumer_dispute_for_review() -> None:
    request = ComplaintAnalyzeRequest(problem_text="本人和商户之间是投资加盟经营纠纷，要求退还加盟费。")
    result, review_reasons = ClassifierAgent().classify(request)

    assert result.accept_suggestion == AcceptSuggestion.REVIEW
    assert result.reason_type == ReasonType.ARTICLE16_3_NOT_CONSUMER_DISPUTE
    assert review_reasons


def test_classifier_marks_missing_materials_for_review() -> None:
    request = ComplaintAnalyzeRequest(problem_text="买到商品有问题，但是商家不详，也没有凭证。")
    result, review_reasons = ClassifierAgent().classify(request)

    assert result.accept_suggestion == AcceptSuggestion.REVIEW
    assert result.reason_type == ReasonType.ARTICLE16_5_MISSING_OR_FALSE_MATERIALS
    assert review_reasons


def test_classifier_marks_veterinary_product_as_out_of_scope() -> None:
    request = ComplaintAnalyzeRequest(
        problem_text=(
            "市民于5月份在青铜峡市甘城子乡宁夏益海供应管理公司，花费1625元购买牛用食品添加剂“氨基丁酸”，"
            "添加剂的作用是防止牛打架，食用后牛依旧兴奋，要求商家退货赔偿。"
        )
    )
    result, review_reasons = ClassifierAgent().classify(request)

    assert result.is_market is False
    assert result.accept_suggestion == AcceptSuggestion.REVIEW
    assert result.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    assert "suggest_department=农业农村局" in result.evidence_fields
    assert review_reasons


def test_reply_suggests_agriculture_department_for_veterinary_product() -> None:
    request = ComplaintAnalyzeRequest(problem_text="购买牛用食品添加剂后认为存在问题，要求退货赔偿。")
    classification, _ = ClassifierAgent().classify(request)
    reply = ReplyAgent().draft(classification, None, [])

    assert "建议您向农业农村局反映" in reply.text


def test_retrieval_finds_housing_basis_for_property_fee() -> None:
    hits = RetrievalAgent().retrieve(
        ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        "小区物业费和停车位产权纠纷，要求处理。",
    )

    assert hits
    assert hits[0].suggested_department == "住建部门"
    assert "物业" in hits[0].title


def test_retrieval_finds_agriculture_basis_for_veterinary_product() -> None:
    hits = RetrievalAgent().retrieve(
        ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        "购买牛用食品添加剂后认为存在问题，要求退货赔偿。",
    )

    assert hits
    assert hits[0].suggested_department == "农业农村局"
    assert "兽" in hits[0].title or "农业" in hits[0].title


def test_reply_uses_rag_department_when_classifier_has_no_department() -> None:
    classification = ClassificationResult(
        is_market=False,
        accept_suggestion=AcceptSuggestion.REVIEW,
        reason_type=ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        confidence=0.88,
        decision_source=DecisionSource.RULE,
        evidence_fields=["物业费"],
    )
    hits = RetrievalAgent().retrieve(classification.reason_type, "物业费纠纷")
    reply = ReplyAgent().draft(classification, None, hits)

    assert "建议您向住建部门反映" in reply.text
    assert "依据参考：" in reply.text


def test_retrieval_finds_marked_price_rules_for_marked_price() -> None:
    hits = RetrievalAgent().retrieve(
        ReasonType.ARTICLE16_7_OTHER_LEGAL_REASONS,
        "商家没有明码标价，多收费用。",
    )

    assert hits
    assert "明码标价和禁止价格欺诈规定" in hits[0].title


def test_retrieval_finds_consumer_law_for_seven_day_return() -> None:
    hits = RetrievalAgent().retrieve(
        ReasonType.ARTICLE16_7_OTHER_LEGAL_REASONS,
        "网购商品七日内要求退货退款。",
    )

    assert hits
    assert hits[0].title in {"中华人民共和国消费者权益保护法 第二十四条", "中华人民共和国消费者权益保护法 第二十五条"}


def test_retrieval_includes_complaint_rule_for_reject_acceptance() -> None:
    hits = RetrievalAgent().retrieve(
        ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        "投诉事项不属于市场监管职责，不予受理。",
    )

    assert any("市场监督管理投诉举报处理办法" in hit.title and "第十六条" in hit.title for hit in hits)


def test_retrieval_finds_current_complaint_rule_for_reject_acceptance() -> None:
    hits = RetrievalAgent().retrieve(
        ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY,
        "投诉事项不属于市场监管职责，不予受理。",
        top_k=5,
    )

    assert any("市场监督管理投诉举报处理办法" in hit.title and "第十六条" in hit.title for hit in hits)


def test_retrieval_finds_food_safety_law_for_spoiled_food() -> None:
    hits = RetrievalAgent().retrieve(
        ReasonType.ARTICLE16_7_OTHER_LEGAL_REASONS,
        "餐馆食品变质过期，要求处理。",
    )

    assert hits
    assert "中华人民共和国食品安全法" in hits[0].title


def test_retrieval_finds_marked_price_fraud_rules() -> None:
    hits = RetrievalAgent().retrieve(
        ReasonType.ARTICLE16_7_OTHER_LEGAL_REASONS,
        "商家明码标价不清并存在价格欺诈。",
    )

    assert hits
    assert "明码标价和禁止价格欺诈规定" in hits[0].title


def test_retrieval_records_chroma_or_fallback_status() -> None:
    agent = RetrievalAgent()
    hits = agent.retrieve(
        ReasonType.ARTICLE16_7_OTHER_LEGAL_REASONS,
        "餐馆食品变质过期，要求处理。",
    )

    assert hits
    assert agent.last_status["retrieval_source"] in {"CHROMA", "RULE_FALLBACK"}
    assert agent.last_status["hit_count"] == len(hits)


def test_classifier_accepts_fitness_membership_refund_dispute() -> None:
    request = ComplaintAnalyzeRequest(
        problem_text=(
            "商户名称：“金石健身俱乐部”商户地址：青铜峡市新百CCMALL四楼发生时间：2025年8月27日"
            "消费金额：800元问题描述：市民在该健身俱乐部办理年卡会员卡后，发现个人独立柜子被他人使用，"
            "随后与店铺工作人员沟通会员卡退费事宜，却被告知若退费需扣除1个月费用。"
            "主要诉求：请求相关部门协调解决会员卡费用纠纷问题。"
        )
    )

    result, review_reasons = ClassifierAgent().classify(request)

    assert result.is_market is True
    assert result.accept_suggestion == AcceptSuggestion.ACCEPT
    assert result.reason_type == ReasonType.UNKNOWN
    assert result.decision_source == DecisionSource.RULE
    assert "健身" in result.evidence_fields
    assert not review_reasons


def test_classifier_marks_agricultural_pesticide_as_out_of_scope() -> None:
    request = ComplaintAnalyzeRequest(problem_text="购买蚊蝇香后发现可能属于农药产品，要求处理。")
    result, _ = ClassifierAgent().classify(request)

    assert result.is_market is False
    assert result.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    assert "suggest_department=农业农村局" in result.evidence_fields


def test_classifier_marks_property_fee_as_housing_department() -> None:
    request = ComplaintAnalyzeRequest(problem_text="自行车棚管理收费不退，属于物业管理费用问题。")
    result, _ = ClassifierAgent().classify(request)

    assert result.is_market is False
    assert result.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    assert "suggest_department=住建部门" in result.evidence_fields


def test_classifier_marks_public_security_dispute_as_out_of_scope() -> None:
    request = ComplaintAnalyzeRequest(problem_text="按摩店技师操作失误后双方打架，已经报警由派出所处理。")
    result, _ = ClassifierAgent().classify(request)

    assert result.is_market is False
    assert result.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    assert "suggest_department=公安机关" in result.evidence_fields


def test_dispatch_aluminum_factory_to_hexi() -> None:
    request = ComplaintAnalyzeRequest(problem_text="食品质量问题", incident_location="青铜峡市铝厂附近")
    result = DispatchAgent().dispatch(request)

    assert result.office_name == "河西市场监管所"
    assert result.needs_review is False


def test_dispatch_extracts_address_from_problem_text() -> None:
    request = ComplaintAnalyzeRequest(
        problem_text="商户名称：某超市 商户详细位置：青铜峡市铝厂附近 问题描述：购买食品过期。"
    )
    result = DispatchAgent().dispatch(request)

    assert result.office_name == "河西市场监管所"
    assert result.matched_rule in {"铝厂", "青铜峡市铝厂"}


def test_dispatch_cuts_structured_problem_fields_and_uses_clean_manual_rule() -> None:
    request = ComplaintAnalyzeRequest(
        problem_text=(
            "商户名称：火辣辣饭店商户地址：青铜峡市小坝镇发生时间：8月30日消费金额：100元"
            "问题描述：市民在该店就餐，发现蔬菜上有虫，火锅丸子变质有臭味，向商家反映不予处理"
            "主要诉求：协调处理该店蔬菜有虫及丸子变质的问题。"
        )
    )

    result = DispatchAgent().dispatch(request)

    assert result.office_name == "裕民市场监管所"
    assert result.matched_rule == "火辣辣饭店"


def test_dispatch_manual_rules_have_highest_priority() -> None:
    request = ComplaintAnalyzeRequest(problem_text="测试", enterprise_address="青铜峡铝厂生活区")
    result = DispatchAgent().dispatch(request)

    assert result.office_name == "河西市场监管所"
    assert result.matched_rule == "青铜峡铝厂生活区"


def test_dispatch_drug_complaint_to_bureau() -> None:
    request = ComplaintAnalyzeRequest(
        problem_text="市民在青铜峡市小坝镇某药店购买感冒药，认为药品存在质量问题，要求处理。"
    )
    result = DispatchAgent().dispatch(request)

    assert result.office_code == "QTX_BUREAU"
    assert result.office_name == "青铜峡市市场监督管理局"
    assert result.matched_rule == "药品事项:药品"
    assert result.needs_review is False


def test_dispatch_veterinary_drug_does_not_use_drug_bureau_rule() -> None:
    request = ComplaintAnalyzeRequest(
        problem_text="市民购买牛用兽药后认为存在问题，要求退货。",
        enterprise_address="青铜峡市铝厂附近",
    )
    result = DispatchAgent().dispatch(request)

    assert result.office_name == "河西市场监管所"
    assert not str(result.matched_rule).startswith("药品事项")


def test_classifier_marks_irrelevant_input_for_review() -> None:
    request = ComplaintAnalyzeRequest(problem_text="你好")
    result, review_reasons = ClassifierAgent().classify(request)

    assert result.accept_suggestion == AcceptSuggestion.REVIEW
    assert "invalid_input" in result.evidence_fields
    assert review_reasons


def test_reject_reason_agent_uses_rule_reason_first() -> None:
    request = ComplaintAnalyzeRequest(problem_text="购买牛用食品添加剂后认为存在问题，要求退货赔偿。")
    classification, _ = ClassifierAgent().classify(request)
    suggestion = RejectReasonAgent().suggest(request, classification)

    assert suggestion is not None
    assert suggestion.reason_type == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    assert suggestion.decision_source.value == "RULE"


def test_reject_reason_agent_model_fallback_for_unknown_review() -> None:
    request = ComplaintAnalyzeRequest(problem_text="商家拒绝退款，要求协调处理。")
    classification, _ = ClassifierAgent().classify(request)
    suggestion = RejectReasonAgent().suggest(request, classification)

    assert suggestion is not None
    assert suggestion.decision_source in {DecisionSource.MODEL, DecisionSource.FALLBACK}
