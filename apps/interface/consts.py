"""Constants used throughout the interface."""

APP_ENC = "encounter"
APP_DIAG = "diagnosis"
APP_EVNT = "event"

APP_PAGE_QUERY = "query"
APP_PAGE_ANALYZE = "analyze"
APP_PAGE_VISUALIZE = "visualize"

TABLES = ["Encounters", "Diagnoses", "Events"]
TABLE_IDS = [APP_ENC, APP_DIAG, APP_EVNT]

NAV_PAGE_IDS = [APP_PAGE_QUERY, APP_PAGE_ANALYZE, APP_PAGE_VISUALIZE]
NAV_PAGES = [page_id.capitalize() for page_id in NAV_PAGE_IDS]
NAV_PAGE_BUTTON_GRADIENTS = {
    APP_PAGE_QUERY.capitalize(): {"from": "teal", "to": "lime", "deg": 105},
    APP_PAGE_ANALYZE.capitalize(): {"from": "indigo", "to": "cyan", "deg": 105},
    APP_PAGE_VISUALIZE.capitalize(): {"from": "grape", "to": "pink", "deg": 105},
}
NAV_PAGE_BUTTON_ICONS = {
    APP_PAGE_QUERY.capitalize(): "medical-icon:administration",
    APP_PAGE_ANALYZE.capitalize(): "icon-park:analysis",
    APP_PAGE_VISUALIZE.capitalize(): "carbon:time-plot",
}

CACHE_TIMEOUT = 3000
STATIC = "static"
TEMPORAL = "temporal"

EVALUATION = "evaluation"
TIMELINE = "timeline"
FEATURE_STORE = "feature_store"
DIRECT_LOAD = "direct_load"
