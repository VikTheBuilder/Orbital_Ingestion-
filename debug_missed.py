import sys
sys.stdout.reconfigure(encoding='utf-8')
from backend.ingestion.text_normalizer import split_into_obligation_units, is_boilerplate, is_quoted_content
from backend.core.rule_engine import get_rule_engine
from backend.ingestion.candidate_engine import generate_candidates, assemble_obligation

re_engine = get_rule_engine()

sections = {
    '121A': (
        "121A. A bank may operate their calamity affected branches from temporary\n"
        "premises under advice to the concerned Regional Office of RBI. For continuing\n"
        "the temporary premise beyond 30 days, banks may obtain specific approval\n"
        "from the respective Regional Off ice of RBI. A bank shall also make\n"
        "arrangements to render banking services in the affected areas by setting up\n"
        "satellite offices, extension counters or mobile banking facilities etc. under\n"
        "intimation to Reserve Bank.\n\n2"
    ),
    '121B': (
        "121B. A bank shall take immediate action for restoration of ATM services at the\n"
        "earliest. During the period, it shall provide alternative arrangements to address\n"
        "the immediate cash requirements of the affected areas."
    ),
    '121C': (
        "121C. Persons displaced or adversely affected by a calamity may not have\n"
        "access to their identification and personal records. In such cases, small\n"
        "accounts as stipulated in the Reserve Bank of India (Commercial Banks- Know\n"
        "Your Customer) Directions, 2025, may be opened by banks."
    ),
    '121D': (
        "121D. A bank at its discretion, may provide relief measures such as waiver /\n"
        "reduction of various fees and charges in respect of customers in the areas\n"
        "where a calamity has been declared, for a period not exceeding one year."
    ),
}

for sec_id, text in sections.items():
    units = split_into_obligation_units(text, 'obligation')
    print(f"\n=== {sec_id}: {len(units)} units ===")
    for i, u in enumerate(units, 1):
        boiler = is_boilerplate(u)
        quoted = is_quoted_content(u, 'obligation')
        print(f"  unit {i}: boiler={boiler} quoted={quoted}")
        print(f"    '{u[:100]}'")
        if boiler or quoted:
            print("    -> SKIPPED")
            continue
        bundle = generate_candidates(u, f'{sec_id}-U{i}', sec_id, sec_id, 'obligation', re_engine, 'RBI')
        ob = assemble_obligation(bundle, [], re_engine)
        if ob:
            print(f"    -> conf={ob.confidence:.3f} type={ob.obligation_type}")
            print(f"       action: {ob.action[:100]}")
        else:
            # Find why it was dropped
            from backend.ingestion.candidate_engine import aggregate_field
            tv = aggregate_field(bundle.candidates, 'trigger')
            dv = aggregate_field(bundle.candidates, 'deadline')
            xv = aggregate_field(bundle.candidates, 'cross_references')
            trigger_rule = tv.votes[0].metadata.get('raw_rule') if tv and tv.votes else None
            dl_urgency = dv.value.get('urgency', 'ongoing') if dv else 'ongoing'
            has_xref = bool(xv and xv.value)
            print(f"    -> DROPPED: trigger={trigger_rule is not None} deadline_urgency={dl_urgency} xref={has_xref}")
            print(f"       clause_type check: trigger_weight={trigger_rule.get('weight',0) if trigger_rule else 0}")
