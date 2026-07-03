from app.core.enums import ReasonType
from app.tools.build_reject_reason_candidates import classify_reject_reason


def test_classify_reject_reason_agriculture_department() -> None:
    result = classify_reject_reason(
        "购买牛用食品添加剂后认为存在问题，要求退货赔偿。",
        "兽药及兽用产品由农业农村局监管，建议由农业农村局负责核实处理。",
    )

    assert result["reason_type"] == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    assert result["suggested_department"] == "农业农村局"
    assert result["trainable"] is True


def test_classify_reject_reason_post_process_not_trainable() -> None:
    result = classify_reject_reason("在超市购买食品发现过期。", "投诉人已撤诉。")

    assert result["reason_type"] == "UNKNOWN"
    assert result["trainable"] is False
    assert result["needs_manual_review"] is True


def test_classify_reject_reason_non_consumer_dispute() -> None:
    result = classify_reject_reason("购买硅铁用于经营，要求退还货款。", "该消费为经营性消费，不属于生活消费。")

    assert result["reason_type"] == ReasonType.ARTICLE16_3_NOT_CONSUMER_DISPUTE
    assert result["trainable"] is True

def test_classify_reject_reason_housing_department() -> None:
    result = classify_reject_reason(
        "自行车棚管理收费不退，属于物业管理问题。",
        "自行车棚管理和收费是属于物业管理方面的问题，由住建局物监办管辖。",
    )

    assert result["reason_type"] == ReasonType.ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY
    assert result["suggested_department"] == "住建部门"


def test_classify_reject_reason_missing_materials() -> None:
    result = classify_reject_reason(
        "买到商品有问题，但是商家不详，也没有凭证。",
        "被投诉人主体不明，无法提供消费凭证。",
    )

    assert result["reason_type"] == ReasonType.ARTICLE16_5_MISSING_OR_FALSE_MATERIALS
