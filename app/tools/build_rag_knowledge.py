from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


CHINESE_NUMS = set("一二三四五六七八九十百零〇两")
ARTICLE_PREFIX = "\u7b2c"
ARTICLE_SUFFIX = "\u6761"

SOURCES = [
    {
        "doc_id": "complaint_report_rules",
        "title": "市场监督管理投诉举报处理办法",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2026/art_e4d03a20c0fd49769e408c7bf3791ff5.html",
        "source": "国家市场监督管理总局",
        "status": "现行有效",
        "start_contains": "市场监督管理投诉举报处理办法",
        "start_occurrence": 2,
        "keywords": ["投诉", "举报", "不予受理", "受理", "调解", "消费者权益争议", "第十六条", "处理权限", "真实身份信息"],
        "suggested_department": None,
    },
    {
        "doc_id": "property_fee_rules",
        "title": "物业服务收费管理办法",
        "url": "https://www.gov.cn/gongbao/content/2004/content_62896.htm",
        "source": "中国政府网/国务院公报",
        "status": "现行参考",
        "start_contains": "物业服务收费管理办法",
        "start_occurrence": 3,
        "keywords": ["物业", "物业费", "物业服务", "收费", "明码标价", "公示", "业主", "停车位", "车位", "车棚"],
        "suggested_department": "住建部门",
    },
    {
        "doc_id": "food_safety_law",
        "title": "中华人民共和国食品安全法",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2023/art_6bff4ef87291497fa72949e1fc88efb5.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "中华人民共和国食品安全法",
        "start_occurrence": 4,
        "keywords": ["食品", "食品安全", "餐饮", "过期", "变质", "添加剂", "标签", "生产经营", "召回", "赔偿"],
        "suggested_department": None,
    },
    {
        "doc_id": "product_quality_law",
        "title": "中华人民共和国产品质量法",
        "url": "https://www.samr.gov.cn/zfjcj/tzgg/art/2023/art_579118cd202a45fba28b7edfd9f6fd72.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "中华人民共和国产品质量法",
        "start_occurrence": 2,
        "keywords": ["产品质量", "质量", "缺陷", "合格", "三包", "生产者", "销售者", "赔偿"],
        "suggested_department": None,
    },
    {
        "doc_id": "advertising_law",
        "title": "中华人民共和国广告法",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2023/art_5474cf75173c45d6a0379730fb4e8d97.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "中华人民共和国广告法",
        "start_occurrence": 4,
        "keywords": ["广告", "虚假广告", "虚假宣传", "代言", "互联网广告", "医疗广告", "违法广告"],
        "suggested_department": None,
    },
    {
        "doc_id": "ecommerce_law",
        "title": "中华人民共和国电子商务法",
        "url": "https://www.samr.gov.cn/zfjcj/tzgg/art/2023/art_d337c3291e8b40459ca03dea54395856.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "中华人民共和国电子商务法",
        "start_occurrence": 2,
        "keywords": ["电子商务", "网购", "平台", "网络交易", "平台经营者", "刷单", "评价", "七日退货"],
        "suggested_department": None,
    },
    {
        "doc_id": "anti_unfair_competition_law",
        "title": "中华人民共和国反不正当竞争法",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2023/art_3737890d856a4e44a8ea07c50c90c116.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "中华人民共和国反不正当竞争法",
        "start_occurrence": 4,
        "keywords": ["不正当竞争", "混淆", "商业贿赂", "虚假宣传", "商业秘密", "有奖销售", "刷单"],
        "suggested_department": None,
    },
    {
        "doc_id": "network_trade_rules",
        "title": "网络交易监督管理办法",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2025/art_4b47c79b8d994a42bba4835997688faa.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "网络交易监督管理办法",
        "start_occurrence": 2,
        "keywords": ["网络交易", "网购", "平台", "直播带货", "网络经营者", "刷单", "评价", "退货"],
        "suggested_department": None,
    },
    {
        "doc_id": "internet_ad_rules",
        "title": "互联网广告管理办法",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2023/art_d93a579afd45413e8576e4623fab348f.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "互联网广告管理办法",
        "start_occurrence": 2,
        "keywords": ["互联网广告", "广告", "弹窗广告", "直播广告", "虚假广告", "可识别性"],
        "suggested_department": None,
    },
    {
        "doc_id": "marked_price_fraud_rules",
        "title": "明码标价和禁止价格欺诈规定",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/fgs/art/2023/art_9a1f82a007964950a1a0f6c056f2fedf.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "明码标价和禁止价格欺诈规定",
        "start_occurrence": 2,
        "keywords": ["明码标价", "价格欺诈", "价格", "收费", "标价", "价外加价", "优惠价", "原价"],
        "suggested_department": None,
    },
    {
        "doc_id": "complaint_report_temp_rules",
        "title": "市场监督管理投诉举报处理暂行办法（已废止，历史参考）",
        "url": "https://www.gov.cn/zhengce/2022-10/08/content_5723506.htm",
        "source": "中国政府网/市场监管总局",
        "status": "已废止，历史参考",
        "start_contains": "市场监督管理投诉举报处理暂行办法",
        "start_occurrence": 1,
        "keywords": ["投诉", "举报", "不予受理", "受理", "调解", "消费者权益争议", "第十五条", "处理权限"],
        "suggested_department": None,
    },
    {
        "doc_id": "price_law",
        "title": "中华人民共和国价格法",
        "url": "https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/jls/art/2023/art_3da9131ab041449d9af2af886ee33766.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "中华人民共和国价格法",
        "start_occurrence": 4,
        "keywords": ["价格", "明码标价", "收费", "价格欺诈", "哄抬价格", "市场调节价", "政府定价", "价格监督检查"],
        "suggested_department": None,
    },
    {
        "doc_id": "consumer_rights_law",
        "title": "中华人民共和国消费者权益保护法",
        "url": "https://www.samr.gov.cn/zfjcj/tzgg/art/2023/art_615af9ed6bcd4974bf853dd2e02bc663.html",
        "source": "国家市场监督管理总局",
        "status": "现行参考",
        "start_contains": "中华人民共和国消费者权益保护法",
        "start_occurrence": 3,
        "keywords": ["消费者", "经营者", "退货", "退款", "赔偿", "七日", "无理由退货", "虚假宣传", "发票", "公平交易"],
        "suggested_department": None,
    },
]

SUPPLEMENTAL_ENTRIES = [
    {
        "id": "transfer_agriculture_veterinary",
        "title": "农业农村、兽药兽用产品、饲料农药类事项",
        "content": "兽药、兽用产品、饲料、饲料添加剂、农药、农机和养殖生产环节事项通常由农业农村部门负责监管处理。",
        "source": "权责清单补充",
        "source_url": "",
        "law_status": "业务规则参考",
        "reason_types": ["ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"],
        "keywords": ["兽药", "兽用", "牛用", "饲料", "饲料添加剂", "农药", "农机", "养殖", "畜牧", "牲畜", "牛打架", "农业农村局"],
        "suggested_department": "农业农村局",
    },
    {
        "id": "transfer_public_security",
        "title": "治安、人身伤害、盗窃报警类事项",
        "content": "打架、人身伤害、盗窃、治安纠纷以及已经报警的事项，通常应由公安机关依法核实处理。",
        "source": "权责清单补充",
        "source_url": "",
        "law_status": "业务规则参考",
        "reason_types": ["ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"],
        "keywords": ["打架", "报警", "派出所", "公安", "盗窃", "人身伤害", "故意伤害", "治安"],
        "suggested_department": "公安机关",
    },
    {
        "id": "transfer_tobacco",
        "title": "烟草专卖类事项",
        "content": "香烟、卷烟、烟草专卖经营许可等事项通常由烟草专卖部门结合职责处理。",
        "source": "权责清单补充",
        "source_url": "",
        "law_status": "业务规则参考",
        "reason_types": ["ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"],
        "keywords": ["香烟", "卷烟", "烟草", "烟草专卖"],
        "suggested_department": "烟草专卖部门",
    },
]


class TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.skip = False

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        data = " ".join(data.split())
        if data:
            self.text.append(data)


def main() -> None:
    entries = []
    summary = {}
    for source in SOURCES:
        articles = split_articles(
            extract_body(
                fetch_text(source["url"]),
                source["start_contains"],
                int(source.get("start_occurrence", -1)),
            )
        )
        if not articles:
            raise RuntimeError(f"No articles parsed for {source['doc_id']}")
        summary[source["doc_id"]] = len(articles)
        for index, (article_no, content) in enumerate(articles, start=1):
            entry = {
                "id": f"{source['doc_id']}_{index:03d}",
                "title": f"{source['title']} {article_no}",
                "content": content,
                "source": source["source"],
                "source_url": source["url"],
                "law_status": source["status"],
                "reason_types": reason_types_for(source["doc_id"], article_no, content),
                "keywords": keywords_for(source["keywords"], article_no, content),
            }
            if source["suggested_department"]:
                entry["suggested_department"] = source["suggested_department"]
            entries.append(entry)
    entries.extend(SUPPLEMENTAL_ENTRIES)

    output_path = Path("data/knowledge/rag_knowledge.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": output_path.as_posix(), "total_entries": len(entries), "by_doc": summary}, ensure_ascii=False, indent=2))


def fetch_text(url: str) -> list[str]:
    raw = urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
    parser = TextParser()
    parser.feed(raw.decode("utf-8", "ignore"))
    return parser.text


def extract_body(items: list[str], start_contains: str, start_occurrence: int = -1) -> list[str]:
    candidates = [index for index, item in enumerate(items) if start_contains in item]
    selected_index = candidates[start_occurrence] if candidates and abs(start_occurrence) < len(candidates) else (candidates[-1] if candidates else -1)
    start = selected_index + 1 if selected_index >= 0 else 0
    body = []
    for item in items[start:]:
        item = normalize_item(item)
        if item in {"相关稿件", "链接："}:
            break
        if item.startswith("主办单位：") or item.startswith("版权所有："):
            break
        if item in {"目 录", "目录"} or item == "第" or is_chapter_line(item):
            continue
        body.append(item)
    return body


def split_articles(items: list[str]) -> list[tuple[str, str]]:
    articles = []
    current_no: str | None = None
    current_parts: list[str] = []
    for item in items:
        if is_article_marker(item):
            if current_no and current_parts:
                articles.append((current_no, "".join(current_parts)))
            current_no = item
            current_parts = []
            continue
        if current_no:
            current_parts.append(item)
    if current_no and current_parts:
        articles.append((current_no, "".join(current_parts)))
    return articles


def normalize_item(text: str) -> str:
    return text.replace("第 ", "第").replace(" 条", "条").strip()


def is_article_marker(item: str) -> bool:
    return len(item) >= 3 and item[0] == ARTICLE_PREFIX and item[-1] == ARTICLE_SUFFIX and all(char in CHINESE_NUMS for char in item[1:-1])


def is_chapter_line(item: str) -> bool:
    return item.startswith("第") and "章" in item and len(item) <= 20


def reason_types_for(doc_id: str, article_no: str, content: str) -> list[str]:
    all_types = [
        "ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY",
        "ARTICLE16_2_ALREADY_ACCEPTED_OR_PROCESSED",
        "ARTICLE16_3_NOT_CONSUMER_DISPUTE",
        "ARTICLE16_4_EXPIRED",
        "ARTICLE16_5_MISSING_OR_FALSE_MATERIALS",
        "ARTICLE16_6_IMPERSONATION_OR_REFUSE_VERIFY",
        "ARTICLE16_7_OTHER_LEGAL_REASONS",
    ]
    if doc_id in {"complaint_report_rules", "complaint_report_temp_rules"}:
        if "不予受理" in content or article_no in {"第十五条", "第十六条"}:
            return all_types
        if any(word in content for word in ["投诉", "举报", "处理权限", "消费者权益争议", "材料"]):
            return all_types
    if doc_id == "property_fee_rules":
        return ["ARTICLE16_1_OUT_OF_SCOPE_OR_NO_AUTHORITY"]
    if doc_id in {
        "price_law",
        "consumer_rights_law",
        "food_safety_law",
        "product_quality_law",
        "advertising_law",
        "ecommerce_law",
        "anti_unfair_competition_law",
        "network_trade_rules",
        "internet_ad_rules",
        "marked_price_fraud_rules",
    }:
        return ["ARTICLE16_7_OTHER_LEGAL_REASONS"]
    return []


def keywords_for(base_keywords: list[str], article_no: str, content: str) -> list[str]:
    keywords = set(base_keywords)
    for keyword in [
        "不予受理",
        "明码标价",
        "价格欺诈",
        "退货",
        "退款",
        "赔偿",
        "七日",
        "发票",
        "物业费",
        "公示",
        "调解",
        "处理权限",
        "消费者权益争议",
        "虚假宣传",
        "食品安全",
        "食品",
        "过期",
        "变质",
        "标签",
        "产品质量",
        "缺陷",
        "广告",
        "虚假广告",
        "网络交易",
        "电子商务",
        "平台",
        "直播带货",
        "不正当竞争",
        "刷单",
        "互联网广告",
        "价格欺诈",
        "价外加价",
    ]:
        if keyword in content:
            keywords.add(keyword)
    keywords.add(article_no)
    return sorted(keywords)


if __name__ == "__main__":
    main()
