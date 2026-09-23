"""Free page-type v8 labels from the schema.org types a page already declares.

Input is `corpus/schema-types.ts` output: per page, the top-level JSON-LD type
counts, the nested ones, and microdata itemtype counts. The page's MAIN entity
is what remains after removing site-wide types, and it maps to one v8 class.

Noise filters, each a named drop reason:
  - site-wide types: a fixed list (Organization, WebSite, BreadcrumbList, ...)
    plus any TEMPLATE-PRONE type (LocalBusiness, FAQPage, ItemList) present on
    >= 60% of a domain's pages (domains with >= 4 pages): a LocalBusiness block
    in a footer, an FAQ widget or a related-items carousel on every page says
    nothing about this one. Only those: a shop's crawl is mostly product pages,
    so Product on 60% of them is the subject of those pages, not chrome;
  - many of one entity: 3+ Products is a listing, 3+ Events or JobPostings an
    index of them;
  - an Article/BlogPosting on the site root is ignored (a homepage feed);
  - two different v8 classes left over = conflict, dropped.

The featurizer never reads schema markup, so these labels train a head that has
never seen their source: schema is hidden on every page by construction.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SITE_WIDE = {
    "Organization", "Corporation", "WebSite", "WebPage", "BreadcrumbList", "ListItem",
    "SearchAction", "EntryPoint", "PropertyValueSpecification", "ImageObject", "VideoObject",
    "SiteNavigationElement", "WPHeader", "WPFooter", "WPSideBar", "Person", "PostalAddress",
    "ContactPoint", "Brand", "Offer", "AggregateOffer", "AggregateRating", "Rating", "Review",
    "Thing", "ReadAction", "SpeakableSpecification", "GeoCoordinates", "OpeningHoursSpecification",
    "Place", "Country", "Language", "CreativeWork", "MonetaryAmount", "QuantitativeValue",
    "PriceSpecification", "UnitPriceSpecification", "OfferShippingDetails", "DefinedRegion",
    "ShippingDeliveryTime", "MerchantReturnPolicy", "PropertyValue", "NewsMediaOrganization",
    "OnlineStore", "OnlineBusiness", "Question", "Answer", "HowToStep", "NutritionInformation",
    "InteractionCounter", "Comment",
}

LOCAL_BUSINESS = {
    "LocalBusiness", "Store", "Restaurant", "CafeOrCoffeeShop", "Bakery", "BarOrPub", "Hotel",
    "LodgingBusiness", "Dentist", "MedicalClinic", "MedicalBusiness", "Physician", "Pharmacy",
    "AutoDealer", "AutoRepair", "AutomotiveBusiness", "BeautySalon", "HealthAndBeautyBusiness",
    "DaySpa", "HairSalon", "ProfessionalService", "LegalService", "Attorney", "RealEstateAgent",
    "HomeAndConstructionBusiness", "Plumber", "Electrician", "HVACBusiness", "GeneralContractor",
    "FoodEstablishment", "ClothingStore", "ElectronicsStore", "FurnitureStore", "GroceryStore",
    "HardwareStore", "JewelryStore", "SportingGoodsStore", "BookStore", "PetStore",
    "FinancialService", "BankOrCreditUnion", "InsuranceAgency", "SportsActivityLocation",
    "ExerciseGym", "EntertainmentBusiness", "TravelAgency", "ChildCare", "Library",
    "EmploymentAgency", "AccountingService",
}
EVENTS = {
    "Event", "MusicEvent", "BusinessEvent", "EducationEvent", "SocialEvent", "SportsEvent",
    "Festival", "TheaterEvent", "ScreeningEvent", "ExhibitionEvent", "ComedyEvent",
    "DanceEvent", "FoodEvent", "LiteraryEvent", "SaleEvent", "CourseInstance", "EventSeries",
}
TYPE_TO_V8 = {
    "Product": "product_detail", "ProductGroup": "product_detail",
    "IndividualProduct": "product_detail", "ProductModel": "product_detail",
    "Article": "article", "NewsArticle": "article", "ReportageNewsArticle": "article",
    "AnalysisNewsArticle": "article", "OpinionNewsArticle": "article", "Report": "article",
    "ScholarlyArticle": "article",
    "BlogPosting": "blog_post", "LiveBlogPosting": "blog_post",
    "TechArticle": "docs", "APIReference": "docs",
    "FAQPage": "faq", "Recipe": "recipe", "JobPosting": "job_posting",
    "AboutPage": "about", "ContactPage": "contact", "SearchResultsPage": "search_results",
    "CollectionPage": "collection", "ItemList": "collection", "OfferCatalog": "collection",
    **{t: "event" for t in EVENTS},
    **{t: "local_business" for t in LOCAL_BUSINESS},
}
PRODUCT_TYPES = {t for t, c in TYPE_TO_V8.items() if c == "product_detail"}
TEMPLATE_PRONE = LOCAL_BUSINESS | {"FAQPage", "ItemList"}
#: v7 classes whose meaning v8 keeps 1:1, so their Jev v7 labels are reused.
V7_REUSED = {"legal", "contact", "about", "homepage", "pricing", "docs"}


def page_label(row: dict, domain_site_wide: set[str]) -> tuple[str | None, str]:
    """(v8 class or None, reason)."""
    ignore = SITE_WIDE | domain_site_wide
    counts: Counter = Counter()
    for type_, n in row["ld"].items():
        if type_ not in ignore:
            counts[type_] += n
    for type_, n in row["md"].items():
        if type_ not in ignore and type_ not in row["ld"]:
            counts[type_] += n
    products = sum(n for t, n in counts.items() if t in PRODUCT_TYPES) + sum(
        n for t, n in row["ldNested"].items() if t in PRODUCT_TYPES
    )
    classes: Counter = Counter()
    for type_, n in counts.items():
        mapped = TYPE_TO_V8.get(type_)
        if mapped:
            classes[mapped] += n
    if not classes:
        all_types = set(row["ld"]) | set(row["md"])
        if row["root"] and all_types & {"Organization", "WebSite", "Corporation"}:
            return "homepage", "root+Organization/WebSite"
        return None, "no main entity"
    if classes.get("product_detail", 0) >= 3:
        classes["product_listing"] = classes.pop("product_detail")
    for many in ("event", "job_posting"):
        if classes.get(many, 0) >= 3:
            classes["listing_index"] = classes.pop(many)
    if "collection" in classes:
        classes.pop("collection")
        target = "product_listing" if products >= 3 else "listing_index"
        classes[target] += 1
    # A blog post often ALSO declares Article; the more specific type wins.
    if "blog_post" in classes and "article" in classes:
        classes.pop("article")
    # A listing page carrying its items' own types (products inside a
    # product_listing) is the listing.
    if "product_listing" in classes and "product_detail" in classes:
        classes.pop("product_detail")
    if row["root"]:
        for editorial in ("article", "blog_post"):
            classes.pop(editorial, None)
        if not classes:
            all_types = set(row["ld"]) | set(row["md"])
            if all_types & {"Organization", "WebSite", "Corporation"}:
                return "homepage", "root+Organization/WebSite (article feed ignored)"
            return None, "article on the site root"
    if len(classes) > 1:
        return None, "conflict"
    return next(iter(classes)), "main entity"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", required=True, help="schema-types.jsonl")
    parser.add_argument("--out", required=True, help="schema-labels-v8.jsonl")
    parser.add_argument("--site-wide-share", type=float, default=0.6)
    parser.add_argument("--min-domain-pages", type=int, default=4)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.types).read_text().splitlines() if line]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[row["d"]].append(row)
    site_wide: dict[str, set[str]] = {}
    for domain, pages in by_domain.items():
        if len(pages) < args.min_domain_pages:
            site_wide[domain] = set()
            continue
        presence: Counter = Counter()
        for page in pages:
            presence.update(set(page["ld"]) | set(page["md"]))
        site_wide[domain] = {
            t for t, n in presence.items()
            if t in TEMPLATE_PRONE and n / len(pages) >= args.site_wide_share
        }

    stats: dict = {"pages": len(rows), "reasons": Counter(), "bySplit": defaultdict(Counter)}
    with Path(args.out).open("w") as handle:
        for row in rows:
            label, reason = page_label(row, site_wide[row["d"]])
            stats["reasons"][reason] += 1
            if label is None:
                continue
            stats["bySplit"][row["s"]][label] += 1
            handle.write(json.dumps({"pageId": row["p"], "split": row["s"], "pageType": label,
                                     "source": "schema", "reason": reason}) + "\n")
    print(json.dumps({"pages": stats["pages"], "reasons": dict(stats["reasons"]),
                      "bySplit": {s: dict(c) for s, c in stats["bySplit"].items()}}, indent=2))


if __name__ == "__main__":
    main()
