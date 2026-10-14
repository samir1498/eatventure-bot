"""Rule tables per state: (category, action, next state). Order = priority
(the first category found is the only one executed in the frame);
next_state None keeps the current state."""

NORMAL = "NORMAL"
RENOVATE = "RENOVATE"
FOOD = "FOOD"
NEW_POINT = "NEW_POINT"
UPGRADE = "UPGRADE"
DELIVERY = "DELIVERY"
HELPER = "HELPER"


# Gray dismiss pair: an unreachable price shows dimmed and
# never resolves on tap — same suppression treatment as up_food.
GRAY_DISMISS_CATEGORIES = ("gray_coin", "gray_max")


# The first four close whatever should not be open,
# before any game action.
#
# up_food sits LAST in NORMAL on purpose: the up_food template (blue
# button w/ coin) also matches the main-view cost buttons at ~0.97
# confidence. At high priority it hijacks NORMAL and fires 4s
# upgrade_food presses on the wrong screen, overlapping the FOOD
# panel cycle. Demoted below food/box so real station taps win; the
# FOOD state still handles the panel.
#
# food prefers badged dishes: dishes WITH a red upgrade badge are
# upgradeable, so the bot taps the badged one first but still taps a
# badgeless dish when no badge is detected (badge templates miss on
# some views; a strict gate blinded the bot entirely). With neither
# dish nor badge, the orphan-badge fallback taps the badge center
# as a proxy (see _find_fallback_proxy).
# The gray pair (dimmed price chips) usually sits on the MAIN view as
# permanent chrome for unaffordable/maxed stations — a tap never
# resolves them. So they sit BELOW the real game actions: if they
# outranked plane/build/delivery/new_point/food, an unaffordable chip
# would starve the whole city (315 gray_max taps in one 5.5h run).
# The GRAY_STUCK suppression still protects panels where they truly
# block.
NORMAL_RULES = [
    ("open_store", "open_store_click", None),
    ("close", "close", None),
    ("box", "open_box", None),

    ("plane", "plane", RENOVATE),
    ("build", "click", RENOVATE),
    ("upgrade", "upgrade", UPGRADE),
    ("delivery", "delivery", DELIVERY),
    ("new_point", "new_point", NEW_POINT),
    ("food", "food", FOOD),
    ("up_food", "upgrade_food", None),

    ("gray_max", "gray_max", None),
    ("gray_coin", "gray_coin", None),
]

RENOVATE_RULES = [
    ("renovate", "renovate_click", NORMAL),
    ("fly", "renovate_click", NORMAL),
]

DELIVERY_RULES = [
    ("accept_delivery", "accept_delivery", NORMAL),
    ("helper", "helper", HELPER),
    ("close", "close", NORMAL),
]

HELPER_RULES = [
    ("invite_helper", "invite_helper", NORMAL),
    ("delivery", "delivery", DELIVERY),
    ("close", "close", NORMAL),
]

# up_upgrade stays in UPGRADE on purpose: on the next
# frame we look for another item to upgrade. We only leave the
# screen when there is nothing left.
UPGRADE_RULES = [
    ("up_upgrade", "upgrade_item", None),
    ("close", "close", NORMAL),
]

STATE_RULES = {
    NORMAL: NORMAL_RULES,
    RENOVATE: RENOVATE_RULES,
    DELIVERY: DELIVERY_RULES,
    HELPER: HELPER_RULES,
    UPGRADE: UPGRADE_RULES,

    # FOOD and NEW_POINT have their own wait after the press,
    # so they are handled in a dedicated handler. The categories
    # stay here only for the detector filter.
    FOOD: [
        ("up_food", "upgrade_food", None),
        ("gray_max", "gray_max", None),
        ("gray_coin", "gray_coin", None)
    ],
    NEW_POINT: [
        ("up_food", "new_point_click", None),
        ("gray_max", "gray_max", None),
        ("gray_coin", "gray_coin", None),
    ],
}
