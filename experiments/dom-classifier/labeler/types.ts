/**
 * Versioned, additive DOM taxonomy. v1-v3 records retain their original
 * canonical strings; v2 only adds labels and never projects an old label to a
 * new one. An omitted axis is unobserved, while `unknown` is an explicit,
 * singleton observation for that axis.
 */
export const TAXONOMY_REVISION = "dom-taxonomy-v2" as const;

export const ROLES = [
  "site_header",
  "footer",
  "navigation",
  "main_content",
  "article_header",
  "card",
  "aside",
  "form",
  "consent_banner",
  "unknown",
] as const;

export const CONTEXTS = ["site", "article", "main", "header", "footer", "unknown"] as const;
/** Where a component sits in the page. Multiple regions are permitted. */
export const REGIONS = [
  "site_header",
  "footer",
  "sidebar",
  "main_content",
  "article_header",
  "hero",
  "top_banner",
  "bottom_banner",
  "overlay",
  "article_body",
  "advertisement",
  "product_gallery",
  "product_buy_box",
  "product_details",
  "product_reviews",
  "author_bio",
  "related_content",
  "comments",
  "unknown",
] as const;
/** What a component does. Multiple functions are permitted. */
export const FUNCTIONS = ["navigation", "card", "form", "consent_banner", "unknown"] as const;
/** v3: the observed component shape, independent of layout and purpose. */
export const COMPONENT_TYPES = [
  "button",
  "link",
  "input",
  "checkbox",
  "radio",
  "select",
  "toggle",
  "heading",
  "text",
  "icon",
  "image",
  "search",
  "dropdown",
  "tabs",
  "accordion",
  "pagination",
  "card",
  "form",
  "list",
  "table",
  "article",
  "media",
  "hero",
  "navigation_menu",
  "content_section",
  "layout_container",
  "dialog",
  "drawer",
  "tooltip",
  "notification",
  "banner",
  "popover",
  "ad_unit",
  "media_gallery",
  "purchase_panel",
  "specification_list",
  "rating_summary",
  "review_list",
  "video_player",
  "audio_player",
  "author_card",
  "comment_thread",
  "unknown",
] as const;
/** v3: user-observed purpose; this intentionally does not describe source code. */
export const PURPOSES = [
  "navigation",
  "submit",
  "search",
  "filter",
  "consent",
  "share",
  "account",
  "dismiss",
  "download",
  "toggle",
  "promotion",
  "announcement",
  "authentication",
  "subscription",
  "feedback",
  "support",
  "advertising",
  "purchase",
  "media_playback",
  "review",
  "information",
  "editorial",
  "instruction",
  "product_information",
  "comparison",
  "social_proof",
  "unknown",
] as const;
export const OBSERVED_STATES = [
  "visible",
  "disabled",
  "expanded",
  "selected",
  "open",
  "sticky",
  "unknown",
] as const;
/** Page-level review taxonomy. It is deliberately separate from DOM labels. */
export const PAGE_TYPES = [
  "homepage",
  "landing_page",
  "campaign",
  "about",
  "team",
  "contact",
  "location",
  "pricing",
  "product_detail",
  "product_listing",
  "features",
  "service",
  "solution",
  "use_case",
  "industry",
  "integrations",
  "integration_detail",
  "download",
  "docs_index",
  "docs_article",
  "tutorial",
  "api_reference",
  "blog_index",
  "blog_post",
  "news_index",
  "news_article",
  "category",
  "tag",
  "archive",
  "author",
  "changelog",
  "releases",
  "release_notes",
  "careers",
  "job",
  "event_index",
  "event_detail",
  "webinar",
  "video",
  "gallery",
  "community",
  "forum_thread",
  "search_results",
  "account",
  "login",
  "signup",
  "checkout",
  "cart",
  "confirmation",
  "legal",
  "privacy",
  "terms",
  "policy",
  "accessibility",
  "support",
  "faq",
  "status",
  "maintenance",
  "portfolio",
  "case_study",
  "comparison",
  "review",
  "directory",
  "profile",
  "dashboard",
  "error",
  "other",
  "unknown",
] as const;
export const CONTENT_KINDS = [
  "article",
  "product",
  "person",
  "event",
  "review",
  "offer",
  "job",
  "software",
  "service",
  "organization",
  "reference",
  "media",
  "other",
  "unknown",
] as const;
/** Authoritative picker grouping for page types; clients should not duplicate it. */
export const PAGE_TYPE_GROUPS = [
  {
    id: "entry",
    label: "Entry and company",
    values: ["homepage", "landing_page", "campaign", "about", "team", "contact", "location"],
  },
  {
    id: "commercial",
    label: "Commercial",
    values: [
      "pricing",
      "product_detail",
      "product_listing",
      "features",
      "service",
      "solution",
      "use_case",
      "industry",
      "integrations",
      "integration_detail",
      "download",
      "comparison",
      "review",
    ],
  },
  {
    id: "documentation",
    label: "Documentation and releases",
    values: [
      "docs_index",
      "docs_article",
      "tutorial",
      "api_reference",
      "changelog",
      "releases",
      "release_notes",
    ],
  },
  {
    id: "publishing",
    label: "Publishing and media",
    values: [
      "blog_index",
      "blog_post",
      "news_index",
      "news_article",
      "category",
      "tag",
      "archive",
      "author",
      "portfolio",
      "case_study",
      "video",
      "gallery",
    ],
  },
  {
    id: "community",
    label: "Community and events",
    values: [
      "careers",
      "job",
      "event_index",
      "event_detail",
      "webinar",
      "community",
      "forum_thread",
      "directory",
      "profile",
    ],
  },
  {
    id: "transaction",
    label: "Account and transaction",
    values: ["account", "login", "signup", "checkout", "cart", "confirmation", "dashboard"],
  },
  {
    id: "utility",
    label: "Utility and policy",
    values: [
      "search_results",
      "legal",
      "privacy",
      "terms",
      "policy",
      "accessibility",
      "support",
      "faq",
      "status",
      "maintenance",
      "error",
      "other",
      "unknown",
    ],
  },
] as const satisfies ReadonlyArray<{
  id: string;
  label: string;
  values: readonly (typeof PAGE_TYPES)[number][];
}>;
export const COMPONENT_SUBTYPES: Partial<
  Record<(typeof COMPONENT_TYPES)[number], readonly string[]>
> = {
  button: ["primary", "secondary", "icon"],
  input: ["text", "search", "email", "password", "number"],
  card: ["product", "article", "result"],
  navigation_menu: ["main", "footer", "breadcrumb"],
  form: ["search", "sign_in", "contact", "newsletter", "checkout"],
  /** This describes modal behavior, while observedState records whether it is currently open. */
  dialog: ["modal", "non_modal"],
  banner: ["announcement", "promotional", "cookie_consent"],
  ad_unit: ["display", "sponsored_content"],
  media_gallery: ["product", "editorial", "portfolio"],
  purchase_panel: ["product", "subscription"],
  drawer: ["navigation", "cart", "preferences"],
  popover: ["menu", "help"],
  content_section: [
    "features",
    "benefits",
    "pricing",
    "testimonials",
    "faq",
    "logo_cloud",
    "team",
    "contact",
    "related_content",
    "call_to_action",
  ],
};
export const STATEFUL_COMPONENT_TYPES = {
  visible: ["dialog", "drawer", "tooltip", "notification", "banner", "popover", "dropdown"],
  disabled: ["button", "input", "checkbox", "radio", "select", "toggle"],
  expanded: ["dropdown", "tabs", "accordion"],
  selected: ["tabs", "pagination", "select", "radio", "toggle"],
  open: ["dialog", "drawer", "popover", "dropdown"],
  sticky: ["navigation_menu", "banner"],
} as const;
/**
 * Gold review axes. They ask the exact questions a trained head is scored on,
 * so they are separate from the descriptive v2 taxonomy above rather than a
 * projection of it: `legal` here covers v2's privacy/terms/policy, and v2 has
 * no single `docs` or `listing_index` value. Rows without these axes are
 * unchanged.
 *
 * Page type v8 (policy-v8.1): the 12 classes of the page-type head. The six
 * rare v8 classes (event, search_results, local_business, job_posting, faq,
 * recipe) are `other` here.
 */
export const PAGE_TYPES_V8 = [
  "homepage",
  "about",
  "contact",
  "legal",
  "pricing",
  "blog_post",
  "article",
  "docs",
  "listing_index",
  "product_detail",
  "product_listing",
  "other",
] as const;
/**
 * Section purpose at policy-section-v3 granularity (region level, with the
 * table of contents at its tightest node). recipe_card, event_details and
 * chat_widget fold into `none`, as in the section head.
 */
export const SECTION_PURPOSES_V3 = [
  "faq_block",
  "reviews",
  "breadcrumbs",
  "business_info",
  "product_offer",
  "author_byline",
  "video_embed",
  "ad",
  "cookie_consent",
  "newsletter_signup",
  "social_share",
  "related_content",
  "comments",
  "promo_banner",
  "pagination",
  "table_of_contents",
  "account_login",
  "locale_switcher",
  "none",
] as const;
export const GOLD_POLICY_REVISIONS = {
  pageTypeV8: "dom-suggestions-v8.1",
  sectionPurposeV3: "section-purpose-v3",
} as const;
/** The reviewer's prompt for one class: a concrete yes/no question and the policy one-liner. */
export type GoldQuestion = { label: string; name: string; question: string; definition: string };
export const PAGE_TYPE_V8_QUESTIONS: readonly GoldQuestion[] = [
  {
    label: "homepage",
    name: "Homepage",
    question: "Is this page the site's homepage?",
    definition:
      "The site's root page, or a language/region root such as /de/ or /en-us/: the entry point that introduces the whole site rather than one subject.",
  },
  {
    label: "about",
    name: "About",
    question: "Is this page an About page?",
    definition:
      "Who the organisation or person is: story, mission, team, history, careers overview.",
  },
  {
    label: "contact",
    name: "Contact",
    question: "Is this page a Contact page?",
    definition:
      "How to reach a person: contact or support form, phone, addresses, office hours. A help centre of self-service answers is Docs.",
  },
  {
    label: "legal",
    name: "Legal",
    question: "Is this page a Legal page?",
    definition:
      "Terms, privacy, cookie or accessibility policy, licence or other formal policy text.",
  },
  {
    label: "pricing",
    name: "Pricing",
    question: "Is this page a Pricing page?",
    definition:
      "Plans, tiers or rates presented for comparison, with the intention of choosing one.",
  },
  {
    label: "blog_post",
    name: "Blog post",
    question: "Is this page a blog post?",
    definition:
      "One entry of a blog: an individual, usually dated post under a blog section (/blog/ path, a blog label, or post chrome). When nothing says blog, it is Article.",
  },
  {
    label: "article",
    name: "Article",
    question: "Is this page an Article (not a blog post)?",
    definition:
      "One editorial, news or informational piece read start to end: news story, press release, feature, case study or guide not presented as a blog entry.",
  },
  {
    label: "docs",
    name: "Docs",
    question: "Is this page documentation?",
    definition:
      "Product or API docs, reference, how-to or tutorial written to be consulted while doing something; also a help centre of self-service answers.",
  },
  {
    label: "listing_index",
    name: "Listing (non-product)",
    question: "Is this page an index of other pages that are not products?",
    definition:
      "The page's own content is a list: blog or news index, category or tag archive, docs index, event or job listing, directory.",
  },
  {
    label: "product_detail",
    name: "Product detail",
    question: "Is this page about ONE product, plan or service?",
    definition:
      "One product, plan or service is the page's subject (shop product, software product or feature page, single service page). No price or buy button needed.",
  },
  {
    label: "product_listing",
    name: "Product listing",
    question: "Is this page a listing of MANY products, plans or services?",
    definition:
      "Two or more products, plans or services presented as equals, each linking to its own page: shop category, collection, product search grid, services overview.",
  },
  {
    label: "other",
    name: "Other",
    question: "Is this page Other (none of the 11 specific page types)?",
    definition:
      "Account, login, cart, checkout, error pages, and landing pages NOT about one product or service; also a single event, job, recipe, FAQ page, local-business page or search results.",
  },
];
export const SECTION_PURPOSE_V3_QUESTIONS: readonly GoldQuestion[] = [
  {
    label: "faq_block",
    name: "FAQ block",
    question: "Is the outlined element an FAQ block?",
    definition: "Two or more question-and-answer pairs.",
  },
  {
    label: "reviews",
    name: "Reviews",
    question: "Is the outlined element a reviews or testimonials section?",
    definition: "Customer reviews or testimonials, or the rating summary for them.",
  },
  {
    label: "breadcrumbs",
    name: "Breadcrumbs",
    question: "Is the outlined element a breadcrumb trail?",
    definition: "The trail of links from the site root to this page.",
  },
  {
    label: "business_info",
    name: "Business info",
    question: "Is the outlined element a business info block?",
    definition:
      "A business's postal address, opening hours and/or phone as a block, not a single phone link in a menu.",
  },
  {
    label: "product_offer",
    name: "Product offer",
    question: "Is the outlined element the price and buy control of what this page sells?",
    definition: "The price plus add to cart, buy now or book, for the thing this page sells.",
  },
  {
    label: "author_byline",
    name: "Author byline",
    question: "Is the outlined element the author byline of this page's content?",
    definition:
      "The name of the author of this page's content, often with a date. Comment authors and testimonial names are not bylines.",
  },
  {
    label: "video_embed",
    name: "Video embed",
    question: "Is the outlined element an embedded video player?",
    definition: "A video the visitor can play. A muted, autoplaying background video is none.",
  },
  {
    label: "ad",
    name: "Ad (third-party)",
    question: "Is the outlined element a third-party ad?",
    definition:
      "A display, native or sponsored ad, ad slot or ad iframe from an ad network. The site's own promotions are Promo banner.",
  },
  {
    label: "cookie_consent",
    name: "Cookie consent",
    question: "Is the outlined element a cookie consent banner?",
    definition: "A cookie or privacy consent banner or dialog.",
  },
  {
    label: "newsletter_signup",
    name: "Newsletter signup",
    question: "Is the outlined element a newsletter signup?",
    definition: "A form or call to action to subscribe by email.",
  },
  {
    label: "social_share",
    name: "Social share",
    question: "Is the outlined element a set of buttons that share this page?",
    definition:
      "Buttons that share THIS page to social networks, not links to the site's own social profiles.",
  },
  {
    label: "related_content",
    name: "Related content",
    question: "Is the outlined element a related-content list?",
    definition: "Other articles, posts or products offered as related, recommended or read next.",
  },
  {
    label: "comments",
    name: "Comments",
    question: "Is the outlined element a comments section?",
    definition: "The discussion thread under the content, or its reply form.",
  },
  {
    label: "promo_banner",
    name: "Promo banner (own site)",
    question: "Is the outlined element the site's own promo or announcement bar?",
    definition:
      "A site-wide announcement or promotion bar (shipping offer, sale, notice), usually a thin strip at the top. Third-party ads are Ad.",
  },
  {
    label: "pagination",
    name: "Pagination",
    question: "Is the outlined element pagination?",
    definition: "Controls to move between pages of a list (1 2 3 next).",
  },
  {
    label: "table_of_contents",
    name: "Table of contents",
    question: "Is the outlined element this page's table of contents?",
    definition:
      "A list of in-page links to this page's own sections. Yes only for the TIGHTEST element holding the whole list.",
  },
  {
    label: "account_login",
    name: "Account login",
    question: "Is the outlined element a sign-in or registration form or panel?",
    definition: "A sign-in or registration form or panel.",
  },
  {
    label: "locale_switcher",
    name: "Locale switcher",
    question: "Is the outlined element a language, region or currency selector?",
    definition: "A language, region or currency selector.",
  },
  {
    label: "none",
    name: "None",
    question: "Is the outlined element none of the section types?",
    definition:
      "Headers, footers, menus, main content or a page column, or only a fragment of a section (one FAQ question, one review, one breadcrumb link).",
  },
];
/** Shown with every section question: v3 judges the whole region, not the tightest node. */
export const SECTION_V3_GRANULARITY =
  "Yes when the outline's content IS that section: the section, a wrapper around only it, or its main list. A fragment, or a box with substantial unrelated content, is No (none).";
export const BOUNDARIES = ["correct", "too_broad", "too_narrow", "mixed", "unsure"] as const;
export const DECISIONS = ["label", "accept", "reject", "unsure"] as const;

export type Role = (typeof ROLES)[number];
export type Context = (typeof CONTEXTS)[number];
export type Region = (typeof REGIONS)[number];
export type FunctionLabel = (typeof FUNCTIONS)[number];
export type ComponentType = (typeof COMPONENT_TYPES)[number];
export type Purpose = (typeof PURPOSES)[number];
export type ObservedState = (typeof OBSERVED_STATES)[number];
export type PageType = (typeof PAGE_TYPES)[number];
export type ContentKind = (typeof CONTENT_KINDS)[number];
export type PageTypeV8 = (typeof PAGE_TYPES_V8)[number];
export type SectionPurposeV3 = (typeof SECTION_PURPOSES_V3)[number];
/** Which gold axis a suggestion asks about. */
export type GoldAxis = keyof typeof GOLD_POLICY_REVISIONS;
export type ComponentProjection = {
  componentType: ComponentType;
  provenance: "legacy-v1-v2";
};
export type PurposeProjection = {
  purposes: Purpose[];
  provenance: "legacy-v1-v2";
};
export type Boundary = (typeof BOUNDARIES)[number];
export type Decision = (typeof DECISIONS)[number];
export const MODEL_REVIEWS = ["accept", "correct", "reject"] as const;
export type ModelReview = (typeof MODEL_REVIEWS)[number];

/**
 * An extra capture set mounted beside the primary one.
 *
 * A cohort brings its own read-only `captures/` directory and optional
 * `model-suggestions.jsonl`. Labels for every cohort are still appended to the
 * primary data directory's journals, stamped with the cohort id, so mounting a
 * cohort never rewrites or relocates existing labels.
 */
export type CohortSource = { id: string; dir: string };
/** Captures that live in the primary data directory carry this cohort id. */
export const PRIMARY_COHORT_ID = "primary" as const;
/** A cohort id is an opaque short slug, never a path. */
export const COHORT_ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}$/;

export type Rect = { x: number; y: number; width: number; height: number };

export type WeakSuggestion = {
  role: Role;
  provenance: "weak";
  reason: string;
};

export type CapturedNode = {
  id: string;
  parentId: string | null;
  tag: string;
  role: string | null;
  text: string;
  selector: string;
  rect: Rect;
  depth: number;
  suggestion: WeakSuggestion | null;
};

export type CapturedPage = {
  id: string;
  url: string;
  title: string;
  capturedAt: string;
  contentHash: string;
  /** Hash of this immutable rendered capture; currently identical to contentHash. */
  captureHash: string;
  width: number;
  height: number;
  viewport: { width: number; height: number; deviceScaleFactor: number };
  nodes: CapturedNode[];
  screenshotUrl: string;
  split: "training-review";
  /** The source document's full height before the bounded screenshot crop. */
  documentHeight?: number;
  /** True when the capture stopped at the configured full-page height ceiling. */
  heightCapped?: boolean;
  /** Whether the fresh browser render reached a quiet network state before capture. */
  renderSettled?: "networkidle" | "bounded_timeout";
  /** Capture provenance, never a human label. Absent on pre-corpus captures. */
  sourceKind?: "fresh_capture" | "historical_replay";
  /** Original crawl time for a replay, distinct from the rendered `capturedAt`. */
  originalCrawledAt?: string | null;
  /** Original public URL if rendering used a replay or hybrid asset source. */
  sourceUrl?: string | null;
  /** Private opaque corpus reference; it is not an on-disk path. */
  corpusRef?: string | null;
  /** Known content date when distinct from the replay and rendering dates. */
  contentOriginDate?: string | null;
  assetMode?: "live" | "hybrid" | "offline";
  /** Whether a replay ran the stored page's scripts. Absent on fresh captures. */
  scriptMode?: "enabled" | "disabled";
};

export type AnnotationInput = {
  pageId: string;
  nodeId: string;
  decision: Decision;
  role?: Role;
  context?: Context;
  /**
   * Additive v2 labels. Empty means this axis is unlabeled, not negative.
   *
   * Read-only: the store validates and copies every axis (see `checkedAxis`)
   * and never writes back, so a caller may pass a frozen or `as const` array.
   */
  regions?: readonly Region[];
  functions?: readonly FunctionLabel[];
  componentType?: ComponentType;
  componentSubtype?: string;
  /** States seen by the annotator, never inferred by the capture process. */
  observedState?: readonly ObservedState[];
  purposes?: readonly Purpose[];
  /** Gold axis: section purpose at policy-section-v3 granularity. Exactly one value. */
  sectionPurposesV3?: readonly SectionPurposeV3[];
  comment?: string;
  boundary: Boundary;
  clientRequestId: string;
  /** Optional optimistic-concurrency check supplied by a current page response. */
  captureHash?: string;
  supersedes?: string;
  /** Immutable model row the reviewer explicitly acted on, if any. */
  modelSuggestionId?: string;
  /** Human review of that model row; separate from a weak-rule decision. */
  modelReview?: ModelReview;
};

export type Annotation = Omit<
  AnnotationInput,
  | "role"
  | "context"
  | "regions"
  | "functions"
  | "componentType"
  | "componentSubtype"
  | "observedState"
  | "purposes"
  | "sectionPurposesV3"
  | "comment"
  | "captureHash"
  | "supersedes"
> & {
  role: Role | null;
  context: Context | null;
  regions: Region[];
  functions: FunctionLabel[];
  /** v1 records expose compatibility projections; v2 records were labeled on both axes. */
  labelSchemaVersion: 1 | 2;
  componentType: ComponentType | null;
  componentSubtype: string | null;
  observedState: ObservedState[];
  purposes: Purpose[];
  /** Present only on gold-axis answers; rows without it are unchanged. */
  sectionPurposesV3?: SectionPurposeV3[];
  /** 0 means a v1/v2 compatibility projection, not a new human v3 claim. */
  componentSchemaVersion: 0 | 3;
  componentProjection: ComponentProjection | null;
  purposeProjection: PurposeProjection | null;
  comment: string | null;
  captureHash: string;
  supersedes: string | null;
  modelSuggestionId: string | null;
  modelReview: ModelReview | null;
  id: string;
  source: "human";
  gold: false;
  timestamp: string;
  /** Which mounted capture set this label belongs to. Absent on pre-cohort rows. */
  cohortId?: string;
};

export type PageAnnotationInput = {
  pageId: string;
  decision: "label" | "unsure";
  /** Read-only for the same reason as `AnnotationInput`'s axes. */
  pageTypes?: readonly PageType[];
  contentKinds?: readonly ContentKind[];
  /** Gold axis: page type v8 (policy-v8.1). Exactly one value. */
  pageTypesV8?: readonly PageTypeV8[];
  comment?: string;
  clientRequestId: string;
  captureHash?: string;
  supersedes?: string;
  modelSuggestionId?: string;
  modelReview?: ModelReview;
};

export type PageAnnotation = Omit<
  PageAnnotationInput,
  "pageTypes" | "contentKinds" | "pageTypesV8" | "comment" | "captureHash" | "supersedes"
> & {
  id: string;
  pageTypes: PageType[];
  contentKinds: ContentKind[];
  /** Present only on gold-axis answers; rows without it are unchanged. */
  pageTypesV8?: PageTypeV8[];
  comment: string | null;
  captureHash: string;
  supersedes: string | null;
  modelSuggestionId: string | null;
  modelReview: ModelReview | null;
  source: "human";
  gold: false;
  timestamp: string;
  /** Which mounted capture set this label belongs to. Absent on pre-cohort rows. */
  cohortId?: string;
};

/** A standalone rejection preserves feedback without manufacturing a human label. */
export type ModelReviewInput = {
  pageId: string;
  nodeId: string | null;
  modelSuggestionId: string;
  review: "reject";
  comment?: string;
  clientRequestId: string;
  captureHash?: string;
};

export type ModelReviewRecord = Omit<ModelReviewInput, "comment" | "captureHash"> & {
  id: string;
  comment: string | null;
  captureHash: string;
  source: "human";
  timestamp: string;
  /** Which mounted capture set this label belongs to. Absent on pre-cohort rows. */
  cohortId?: string;
};

/** An append-only reversal of one effective human review action. */
export type ReviewActionKind = "annotation" | "page_annotation" | "model_review";
export type ReviewUndoRecord = {
  id: string;
  actionKind: ReviewActionKind;
  actionId: string;
  pageId: string;
  nodeId: string | null;
  captureHash: string;
  timestamp: string;
};

/** A probability from a Noul judgment. It is not a confidence score. */
export type ModelLabelProbability<T extends string> = {
  label: T;
  yesProbability: number;
};

/** A Choice judgment keeps both the selected option and its full distribution. */
export type ModelComponentTypeChoice = {
  choice: ComponentType;
  distribution: Array<{ label: ComponentType; probability: number }>;
  confidence: number;
};

/**
 * Immutable offline model output. This is deliberately separate from captured
 * pages and human annotations: omitted axes are unlabeled, never negative.
 */
export type ModelSuggestion = {
  schemaVersion: 1;
  id: string;
  pageId: string;
  /** Null means this is a page-level prediction rather than a DOM-node prediction. */
  nodeId: string | null;
  captureHash: string;
  provider: "typesafe";
  modelId: string;
  modelRevision: string;
  promptRevision: string;
  /** Present for taxonomy-aware sidecars; omitted legacy sidecars remain valid. */
  taxonomyRevision?: typeof TAXONOMY_REVISION;
  snapshotHash: string;
  rawAnswers: unknown;
  usage?: { inputTokens?: number; outputTokens?: number };
  /** Offline output may be shown, but must never be treated as a human label. */
  provisional: boolean;
  /** Compatibility scalar for a provider's primary raw class, when supplied. */
  rawClass?: string;
  score?: number;
  mappedLabels: {
    regions?: Region[];
    functions?: FunctionLabel[];
    purposes?: Purpose[];
    componentType?: ComponentType;
    pageTypes?: PageType[];
    contentKinds?: ContentKind[];
    /** Gold axes: one value each. A suggestion carrying one is a gold question. */
    pageTypeV8?: PageTypeV8[];
    sectionPurposeV3?: SectionPurposeV3[];
  };
  /**
   * Likely other answers for a gold question, most likely first, offered as
   * one-click fixes after Not OK. Never includes the suggested value.
   */
  /**
   * A blind gold question: the axis is asked with NO suggestion shown, so the
   * reviewer picks the class without an anchor. `mappedLabels` is then empty
   * and there are no alternatives; scoring keys stay private.
   */
  blindAxis?: GoldAxis;
  alternatives?: {
    pageTypeV8?: PageTypeV8[];
    sectionPurposeV3?: SectionPurposeV3[];
  };
  axisProbabilities?: {
    regions?: ModelLabelProbability<Region>[];
    functions?: ModelLabelProbability<FunctionLabel>[];
    purposes?: ModelLabelProbability<Purpose>[];
    pageTypes?: ModelLabelProbability<PageType>[];
    contentKinds?: ModelLabelProbability<ContentKind>[];
  };
  componentTypeChoice?: ModelComponentTypeChoice;
  createdAt: string;
};

/**
 * One gold question in review order (sidecar order across cohorts). `status`
 * is the latest effective human action bound to this suggestion.
 */
export type GoldQueueItem = {
  suggestionId: string;
  pageId: string;
  nodeId: string | null;
  captureHash: string;
  cohortId: string;
  axis: GoldAxis;
  /** The shown class; null on a blind question. */
  label: string | null;
  blind: boolean;
  alternatives: string[];
  /** `unsure` is an explicit "can't tell" bound to the question. */
  status: "pending" | "accepted" | "corrected" | "rejected" | "unsure";
};

/** Read-only projection over the append-only journals for the current captures. */
export type LabelerStats = {
  generatedAt: string;
  pages: {
    total: number;
    /** Pages with a current positive page or element label. */
    labelled: number;
    /** Latest positive page-level classification records, separate from element work. */
    currentPageLabels: number;
  };
  elements: {
    total: number;
    /** Latest `label` or `accept` actions; `reject` and `unsure` are not positive labels. */
    currentHumanLabels: number;
    /** Positive `label` actions with no model suggestion binding. */
    latestManualLabels: number;
  };
  suggestions: {
    total: number;
    page: number;
    element: number;
    pending: number;
    reviewed: { accepted: number; corrected: number; rejected: number };
  };
  /**
   * Per-cohort counts, keyed by cohort id. Always carries the primary cohort so
   * a single-cohort install reads the same as before, plus one entry per mounted
   * cohort. Labels written before cohorts existed count under the primary id.
   */
  byCohort: Record<
    string,
    {
      pages: number;
      labelledPages: number;
      currentHumanLabels: number;
      /**
       * Per annotator, how often a human sided with it or rejected it. This is
       * what makes agreement-with-human comparable between annotators.
       */
      bySource: Record<string, { accepted: number; rejected: number }>;
    }
  >;
};
