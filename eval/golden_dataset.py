"""Golden evaluation dataset for Agent testing.

Usage: from eval.golden_dataset import GOLDEN_SAMPLES, GoldenSample
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GoldenSample:
    id: str
    problem_text: str
    incident_location: str | None = None
    enterprise_address: str | None = None
    expected_is_market: bool | None = None
    expected_accept_suggestion: str | None = None
    expected_reason_type: str | None = None
    expected_office_name: str | None = None
    expected_department: str | None = None


GOLDEN_SAMPLES: list[GoldenSample] = [
    GoldenSample(
        id="eval_001",
        problem_text="在超市购买食品后发现过期，要求退款赔偿。",
        expected_is_market=True,
        expected_accept_suggestion="ACCEPT",
        expected_reason_type="UNKNOWN",
    ),
    GoldenSample(
        id="eval_002",
        problem_text="商户名称：金石健身俱乐部 商户地址：青铜峡市新百CCMALL四楼 消费金额：800元 问题描述：市民在该健身俱乐部办理年卡会员卡后退费纠纷。",
        expected_is_market=True,
        expected_accept_suggestion="ACCEPT",
        expected_reason_type="UNKNOWN",
    ),
    GoldenSample(
        id="eval_003",
        problem_text="市民在餐馆就餐后发现食品安全问题，要求退款。",
        expected_is_market=True,
        expected_accept_suggestion="ACCEPT",
        expected_reason_type="UNKNOWN",
    ),
    GoldenSample(
        id="eval_004",
        problem_text="商家拒绝退款，要求协调处理。",
        expected_is_market=True,
        expected_accept_suggestion="ACCEPT",
        expected_reason_type="UNKNOWN",
    ),
    GoldenSample(
        id="eval_005",
        problem_text="小区物业费和停车位产权纠纷，要求处理。",
        expected_is_market=False,
        expected_accept_suggestion="REVIEW",
        expected_reason_type="ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
        expected_department="住建部门",
    ),
    GoldenSample(
        id="eval_006",
        problem_text="市民于5月份在青铜峡市甘城子乡宁夏益海供应管理公司，花费1625元购买牛用食品添加剂氨基丁酸，添加剂的作用是防止牛打架，食用后牛依旧兴奋，要求商家退货赔偿。",
        expected_is_market=False,
        expected_accept_suggestion="REVIEW",
        expected_reason_type="ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
        expected_department="农业农村局",
    ),
    GoldenSample(
        id="eval_007",
        problem_text="购买蚊蝇香后发现可能属于农药产品，要求处理。",
        expected_is_market=False,
        expected_accept_suggestion="REVIEW",
        expected_reason_type="ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
        expected_department="农业农村局",
    ),
    GoldenSample(
        id="eval_008",
        problem_text="自行车棚管理收费不退，属于物业管理费用问题。",
        expected_is_market=False,
        expected_accept_suggestion="REVIEW",
        expected_reason_type="ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
        expected_department="住建部门",
    ),
    GoldenSample(
        id="eval_009",
        problem_text="买到商品有问题，但是商家不详，也没有凭证。",
        expected_is_market=True,
        expected_accept_suggestion="REVIEW",
        expected_reason_type="ARTICLE16_5_MISSING_OR_FALSE_MATERIALS",
    ),
    GoldenSample(
        id="eval_010",
        problem_text="同一事项法院已受理，消费者再次投诉要求市场监管处理。",
        expected_is_market=True,
        expected_accept_suggestion="REVIEW",
        expected_reason_type="ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED",
    ),
    GoldenSample(
        id="eval_011",
        problem_text="本人和商户之间是投资加盟经营纠纷，要求退还加盟费。",
        expected_is_market=True,
        expected_accept_suggestion="REVIEW",
        expected_reason_type="ARTICLE16_3_NOT_CONSUMER_DISPUTE",
    ),
    GoldenSample(
        id="eval_012",
        problem_text="按摩店技师操作失误后双方打架，已经报警由派出所处理。",
        expected_is_market=False,
        expected_accept_suggestion="REVIEW",
        expected_reason_type="ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
        expected_department="公安机关",
    ),
    GoldenSample(
        id="eval_013",
        problem_text="市民在青铜峡市小坝镇某药店购买感冒药，认为药品存在质量问题，要求处理。",
        expected_is_market=True,
        expected_accept_suggestion="ACCEPT",
        expected_reason_type="UNKNOWN",
        expected_office_name="青铜峡市市场监督管理局",
    ),
    GoldenSample(
        id="eval_014",
        problem_text="你好",
        expected_is_market=True,
        expected_accept_suggestion="REVIEW",
    ),
    GoldenSample(
        id="eval_015",
        problem_text="随便写点",
        expected_is_market=True,
        expected_accept_suggestion="REVIEW",
    ),
]
