# ── 분류 규칙: 인식 결과 → 몇 번 통으로 보낼지 ──
#   통 0~2: config.BIN_FILTERS 에 적힌 분류
#   통 3  : 나머지 전부 (필터에 없는 분류, 토큰/스킬, 인식 확신 없음)
import config

CATEGORY_NAMES = {"monster": "몬스터", "spell": "마법", "trap": "함정", "other": "그 외"}
REJECT_BIN = 3


def category(card_type):
    """DB card_type → 대분류 (monster / spell / trap / other)"""
    card_type = card_type or ""
    if "Monster" in card_type:
        return "monster"
    if card_type == "Spell Card":
        return "spell"
    if card_type == "Trap Card":
        return "trap"
    return "other"   # Token, Skill Card 등


def validate_filters(filters=config.BIN_FILTERS):
    """필터 설정 검사 — 통 3개, 중복 없음, 대분류만"""
    if len(filters) != 3:
        raise ValueError(f"BIN_FILTERS는 3개여야 합니다: {filters}")
    if len(set(filters)) != len(filters):
        raise ValueError(f"BIN_FILTERS에 중복이 있습니다: {filters}")
    unknown = [f for f in filters if f not in ("monster", "spell", "trap")]
    if unknown:
        raise ValueError(f"알 수 없는 필터: {unknown} (monster / spell / trap 중에서)")


def choose_bin(card_type, confident, filters=config.BIN_FILTERS):
    """반환: (통 번호, 대분류)"""
    cat = category(card_type)
    if not confident or cat not in filters:
        return REJECT_BIN, cat
    return filters.index(cat), cat


def bin_label(bin_no, filters=config.BIN_FILTERS):
    if bin_no == REJECT_BIN:
        return f"{REJECT_BIN}번 통 (그 외/확인 필요)"
    return f"{bin_no}번 통 ({CATEGORY_NAMES[filters[bin_no]]})"
