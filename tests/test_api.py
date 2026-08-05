from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_review_workbench_homepage() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "投诉智能处理工作台" in response.text
    assert "执行过程追踪" in response.text
    assert "查看 Trace" in response.text
    assert "复制退回回复" in response.text
    assert "智能问答助手" in response.text
    assert "法规全文" in response.text
    assert "系统运行状态" in response.text


def test_analyze_accept_and_trace() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={
            "complaint_type": "投诉",
            "enterprise_address": "青铜峡市小坝镇步行街",
            "problem_text": "在餐馆就餐后发现食品安全问题，要求退款。",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["is_market"] is True
    assert payload["dispatch"]["office_name"] in {"小坝市场监管所", "裕民市场监管所"}
    assert payload["dispatch"]["decision_source"] == "RULE"
    assert payload["reply_draft"]["template_id"] == "accept_no_return_reply"
    assert payload["reply_draft"]["text"] == ""
    assert payload["agent_steps"]

    trace_response = client.get(f"/api/v1/traces/{payload['trace_id']}")
    assert trace_response.status_code == 200
    assert trace_response.json()["trace_id"] == payload["trace_id"]


def test_accept_path_dispatches_without_return_reply() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={
            "enterprise_address": "青铜峡市XX商场四楼",
            "problem_text": "商户名称：XX健身俱乐部，市民办理年卡会员卡后产生退费纠纷，请求协调处理。",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["is_market"] is True
    assert payload["dispatch"] is not None
    assert payload["reply_draft"]["template_id"] == "accept_no_return_reply"
    assert payload["reply_draft"]["text"] == ""


def test_drug_complaint_dispatches_to_bureau() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={
            "problem_text": "市民在青铜峡市小坝镇某药店购买感冒药，认为药品存在质量问题，要求处理。",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["is_market"] is True
    assert payload["dispatch"]["office_code"] == "QTX_BUREAU"
    assert payload["dispatch"]["office_name"] == "青铜峡市市场监督管理局"
    assert payload["dispatch"]["matched_rule"].startswith("药品事项")


def test_analyze_out_of_scope_review_and_confirm() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={"problem_text": "小区物业费和停车位产权纠纷，要求处理。"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["accept_suggestion"] == "REVIEW"
    assert payload["classification"]["reason_type"] == "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"
    assert payload["reject_reason_suggestion"]["reason_type"] == "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"
    assert payload["reject_reason_suggestion"]["decision_source"] == "RULE"
    assert payload["review_required"] is True
    assert payload["dispatch"] is None

    review_response = client.post(
        f"/api/v1/reviews/{payload['trace_id']}/confirm",
        json={
            "is_market": False,
            "reason_type": "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
            "reject_detail": "物业费用争议建议向住建部门反映。",
            "office_code": "QTX_XIAOBA",
            "office_name": "小坝市场监管所",
            "reply_text": "建议您向住建部门反映。",
            "reviewer": "tester",
        },
    )
    assert review_response.status_code == 200
    assert review_response.json()["saved"] is True


def test_review_stats_endpoint() -> None:
    response = client.get("/api/v1/reviews/stats")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["total"], int)
    assert isinstance(payload["accepted"], int)
    assert isinstance(payload["rejected"], int)
    assert isinstance(payload["unknown"], int)
    assert isinstance(payload["reason_counts"], dict)
    assert isinstance(payload["office_counts"], dict)


def test_system_status_endpoint() -> None:
    response = client.get("/api/v1/system/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["orchestrator_backend"] == "langgraph"
    assert payload["orchestrator_class"] == "LangGraphOrchestrator"
    assert payload["rag"]["knowledge_entries"] >= 700
    assert payload["rag"]["embedding_provider"] in {"hash", "bge", "auto"}
    # 模型文件为可选训练产物（训练脚本可生成），公开仓库不携带——仅校验字段类型
    assert isinstance(payload["models"]["accept_model_exists"], bool)
    assert isinstance(payload["models"]["reject_reason_model_exists"], bool)


def test_export_review_training_endpoint() -> None:
    response = client.post("/api/v1/reviews/export-training")

    assert response.status_code == 200
    payload = response.json()
    assert "accept_training" in payload
    assert "reject_reason_training" in payload
    assert payload["accept_training"]["output_csv"].endswith("accept_training_from_reviews.csv")
    assert payload["reject_reason_training"]["output_csv"].endswith("reject_reason_training_from_reviews.csv")


def test_replay_accept_evaluation_endpoint() -> None:
    response = client.post("/api/v1/evaluation/replay-accept")

    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluated_rows"] > 0
    assert "decision_counts" in payload
    assert payload["outputs"]["errors_csv"].endswith("accept_replay_errors.csv")


def test_mine_accept_rules_endpoint() -> None:
    response = client.post("/api/v1/evaluation/mine-accept-rules")

    assert response.status_code == 200
    payload = response.json()
    assert "candidate_count" in payload
    assert payload["outputs"]["high_accept_csv"].endswith("accept_rule_candidates_high_accept.csv")


def test_simulate_accept_rules_endpoint() -> None:
    response = client.post("/api/v1/evaluation/simulate-accept-rules")

    assert response.status_code == 200
    payload = response.json()
    assert "auto_accept_rows" in payload
    assert payload["outputs"]["false_accept_csv"].endswith("accept_rule_simulation_false_accept.csv")


def test_rag_reject_reply_search_endpoint() -> None:
    response = client.post(
        "/api/v1/rag/reject-reply",
        json={
            "query": "牛用饲料添加剂问题，建议农业农村局处理。",
            "reason_type": "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "reject_reply"
    assert "建议您向农业农村局反映" in payload["reply_suggestion"]
    assert payload["hits"]
    assert any(hit["suggested_department"] == "农业农村局" for hit in payload["hits"])


def test_rag_law_search_endpoint() -> None:
    response = client.post(
        "/api/v1/rag/law-search",
        json={
            "query": "商家没有明码标价，多收费用。",
            "reason_type": "ARTICLE16_7_OTHER_LEGAL_REASONS",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "law_search"
    assert payload["reply_suggestion"] is None
    assert any("明码标价和禁止价格欺诈规定" in hit["title"] for hit in payload["hits"])
    assert all(hit["explanation"] for hit in payload["hits"])


def test_rag_law_documents_endpoint() -> None:
    response = client.get("/api/v1/rag/laws")

    assert response.status_code == 200
    payload = response.json()
    documents = payload["documents"]
    assert any(document["doc_id"] == "consumer_rights_law" for document in documents)
    assert any(document["doc_id"] == "price_law" for document in documents)
    assert any(document["doc_id"] == "complaint_report_rules" for document in documents)
    assert any(document["doc_id"] == "food_safety_law" for document in documents)
    assert any(document["doc_id"] == "advertising_law" for document in documents)


def test_rag_law_full_text_endpoint() -> None:
    response = client.get("/api/v1/rag/laws/consumer_rights_law")

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == "consumer_rights_law"
    assert payload["title"] == "中华人民共和国消费者权益保护法"
    assert len(payload["articles"]) >= 60
    assert payload["articles"][0]["title"] == "中华人民共和国消费者权益保护法 第一条"


def test_rag_current_complaint_rules_full_text_endpoint() -> None:
    response = client.get("/api/v1/rag/laws/complaint_report_rules")

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "市场监督管理投诉举报处理办法"
    assert payload["law_status"] == "现行有效"
    assert any(article["title"] == "市场监督管理投诉举报处理办法 第十六条" for article in payload["articles"])


def test_analyze_irrelevant_input_shows_review_prompt() -> None:
    response = client.post("/api/v1/complaints/analyze", json={"problem_text": "随便写点"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["accept_suggestion"] == "REVIEW"
    assert "invalid_input" in payload["classification"]["evidence_fields"]
    assert payload["dispatch"] is None
    assert "请补充有效投诉内容" in payload["reply_draft"]["text"]


def test_confirm_review_accepts_legacy_reason_type() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={"problem_text": "小区物业费和停车位产权纠纷，要求处理。"},
    )
    payload = response.json()

    review_response = client.post(
        f"/api/v1/reviews/{payload['trace_id']}/confirm",
        json={
            "is_market": False,
            "reason_type": "ARTICLE15_1_OUT_OF_SCOPE",
            "reply_text": "建议您向住建部门反映。",
            "reviewer": "tester",
        },
    )

    assert review_response.status_code == 200
    assert review_response.json()["saved"] is True


def test_confirm_review_saves_article16_reason_type() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={"problem_text": "买到商品有问题，但是商家不详，也没有凭证。"},
    )
    payload = response.json()

    review_response = client.post(
        f"/api/v1/reviews/{payload['trace_id']}/confirm",
        json={
            "is_market": False,
            "reason_type": "ARTICLE16_5_MISSING_OR_FALSE_MATERIALS",
            "reject_detail": "缺少被投诉对象和消费凭证。",
            "office_code": "QTX_YUMIN",
            "office_name": "裕民市场监管所",
            "reply_text": "请补充真实、完整材料后再提交。",
            "reviewer": "tester",
        },
    )

    assert review_response.status_code == 200
    assert review_response.json()["saved"] is True


def test_confirm_review_accepts_reject_detail() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={"problem_text": "购买牛用食品添加剂后认为存在问题，要求退货赔偿。"},
    )
    payload = response.json()

    review_response = client.post(
        f"/api/v1/reviews/{payload['trace_id']}/confirm",
        json={
            "is_market": False,
            "reason_type": "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
            "reject_detail": "兽药及兽用产品由农业农村局监管，建议由农业农村局负责核实处理。",
            "reply_text": "建议您向农业农村局反映。",
            "reviewer": "tester",
        },
    )

    assert review_response.status_code == 200
    assert review_response.json()["saved"] is True


def test_confirm_review_saves_edited_return_reply() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={"problem_text": "小区物业费纠纷，要求市场监管部门处理。"},
    )
    payload = response.json()
    edited_reply = "经核实，该事项建议向住建部门反映。"

    review_response = client.post(
        f"/api/v1/reviews/{payload['trace_id']}/confirm",
        json={
            "is_market": False,
            "reason_type": "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
            "reject_detail": "物业费用争议不属于市场监管职责。",
            "reply_text": edited_reply,
            "reviewer": "tester",
        },
    )

    assert review_response.status_code == 200
    trace_id = payload["trace_id"]
    from app.api.routes import review_store

    saved = review_store.find_by_key("trace_id", trace_id)
    assert saved["review"]["reply_text"] == edited_reply


def test_reject_reason_case_has_no_dispatch_and_reject_reply() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={"problem_text": "买到商品有问题，但是商家不详，也没有凭证。"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["reason_type"] == "ARTICLE16_5_MISSING_OR_FALSE_MATERIALS"
    assert payload["dispatch"] is None
    assert payload["reply_draft"]["template_id"] == "reject_missing_or_false_materials"
    assert "您的投诉已登记" not in payload["reply_draft"]["text"]


def test_unknown_review_case_does_not_show_reject_reason_suggestion() -> None:
    response = client.post(
        "/api/v1/complaints/analyze",
        json={"problem_text": "商家拒绝退款，要求协调处理。"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["classification"]["reason_type"] == "UNKNOWN"
    assert payload["reject_reason_suggestion"] is None
    assert payload["dispatch"] is not None
    assert payload["reply_draft"]["template_id"] == "accept_no_return_reply"
    assert payload["reply_draft"]["text"] == ""
