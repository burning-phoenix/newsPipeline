import unittest
from datetime import datetime

from corroborate.attribution import format_attribution, unverified_attributions
from corroborate.config import PipelineConfig
from corroborate.models import Claim, StoryBrief
from corroborate.stages.rendering import RenderingStage, default_output_filename
from corroborate.stages.story_analysis import StoryAnalysisStage, _truncate_words

OUTLETS = {"BBC News", "Al Jazeera", "France 24", "NPR", "The Guardian"}


class AttributionTest(unittest.TestCase):

    def test_names_all_outlets_up_to_three(self):
        claim = Claim("A bridge collapsed.", {"NPR", "BBC News", "France 24"})
        text = format_attribution(claim)
        self.assertEqual(set(text.strip("()").split(", ")), claim.confirming_outlets)

    def test_more_than_three_are_truncated_with_and_others(self):
        claim = Claim("A bridge collapsed.", set(OUTLETS))
        text = format_attribution(claim)
        self.assertTrue(text.endswith(" and others)"))
        self.assertEqual(len(text[1:-len(" and others)")].split(", ")), 3)

    def test_order_is_stable_and_not_alphabetical_across_claims(self):
        claims = [Claim(f"Claim number {i}.", set(OUTLETS)) for i in range(20)]
        first_named = {format_attribution(c)[1:].split(",")[0] for c in claims}
        # Same input gives the same string ...
        self.assertEqual(format_attribution(claims[0]), format_attribution(claims[0]))
        # ... but across claims, different outlets get named first.
        self.assertGreater(len(first_named), 1)

    def test_flags_parentheticals_that_are_not_verified_attributions(self):
        allowed = {"(BBC News, NPR, France 24)"}
        text = (
            "A bridge collapsed (BBC News, NPR, France 24). "
            "The mayor resigned (Al Jazeera). It was built in 1964 (a landmark)."
        )
        self.assertEqual(
            unverified_attributions(text, allowed, OUTLETS), ["(Al Jazeera)"]
        )

    def test_two_claims_joined_by_semicolon_are_each_checked(self):
        allowed = {"(BBC News, NPR, France 24)", "(The Guardian, NPR and others)"}
        ok = "Two facts (BBC News, NPR, France 24; The Guardian, NPR and others)."
        bad = "Two facts (BBC News, NPR, France 24; The Guardian, Al Jazeera)."
        self.assertEqual(unverified_attributions(ok, allowed, OUTLETS), [])
        self.assertEqual(len(unverified_attributions(bad, allowed, OUTLETS)), 1)


class AnalysisCheckTest(unittest.TestCase):

    def check(self, raw):
        return StoryAnalysisStage(None, PipelineConfig())._check(raw, OUTLETS)

    def test_framing_keeps_known_outlets_once(self):
        notes = self.check({"framing": [
            {"outlet": "BBC News", "angle": "Focused on the victims."},
            {"outlet": "BBC News", "angle": "A second, ignored note."},
            {"outlet": "Made Up Times", "angle": "Not a real outlet here."},
        ]})
        self.assertEqual(notes.framing, {"BBC News": "Focused on the victims."})

    def test_real_conflict_is_kept(self):
        notes = self.check({"discrepancies": [{"topic": "magnitude", "positions": [
            {"outlet": "BBC News", "says": "magnitude 7.1"},
            {"outlet": "NPR", "says": "magnitude 6.8"},
        ]}]})
        self.assertEqual(len(notes.discrepancies), 1)

    def test_omission_is_not_a_conflict(self):
        notes = self.check({"discrepancies": [{"topic": "lifeguard", "positions": [
            {"outlet": "BBC News", "says": "named him as Ryder Williams"},
            {"outlet": "NPR", "says": "did not name the lifeguard"},
        ]}]})
        self.assertEqual(notes.discrepancies, [])

    def test_rounding_is_not_a_conflict(self):
        notes = self.check({"discrepancies": [{"topic": "award", "positions": [
            {"outlet": "BBC News", "says": "$83 million"},
            {"outlet": "NPR", "says": "$83.3 million"},
        ]}]})
        self.assertEqual(notes.discrepancies, [])

    def test_outlet_name_variants_and_map_form_are_accepted(self):
        notes = self.check({"framing": {
            "The Guardian": "Led with the victims.",
            "al jazeera english": "Focused on the region.",
            "The": "Ambiguous: matches more than one outlet, so dropped.",
        }})
        self.assertEqual(set(notes.framing), {"The Guardian", "Al Jazeera"})

    def test_saying_nothing_is_not_a_position(self):
        notes = self.check({"discrepancies": [
            {"topic": "turnout", "positions": [
                {"outlet": "The Guardian", "says": "more than 2.3 million votes"},
                {"outlet": "BBC News", "says": "offered no total"},
            ]},
            {"topic": "weight", "positions": [
                {"outlet": "The Guardian", "says": "an estimated 1,200 pounds"},
                {"outlet": "BBC News", "says": "latest weights had not yet been revealed"},
            ]},
            {"topic": "a first?", "positions": [
                {"outlet": "NPR", "says": "the first time since the referendum"},
                {"outlet": "The Guardian", "says": "does not claim it is the first"},
            ]},
            {"topic": "bipartisan", "positions": [
                {"outlet": "NPR", "says": "passed with bipartisan support"},
                {"outlet": "BBC News", "says": "calls it a Husted bill, implying partisanship"},
            ]},
        ]})
        self.assertEqual(notes.discrepancies, [])

    def test_unique_detail_needs_one_or_two_reporting_outlets(self):
        notes = self.check({"differences": [
            {"kind": "unique", "topic": "lifeguard's name", "positions": [
                {"outlet": "BBC News", "says": "named him as Ryder Williams"},
                {"outlet": "NPR", "says": "did not name the lifeguard"},
            ]},
            {"kind": "unique", "topic": "silence only", "positions": [
                {"outlet": "NPR", "says": "did not report a figure"},
            ]},
            {"kind": "unique", "topic": "too widely reported", "positions": [
                {"outlet": o, "says": "the bridge was built in 1964"}
                for o in ["BBC News", "NPR", "France 24"]
            ]},
        ]})
        self.assertEqual(len(notes.discrepancies), 1)
        self.assertEqual(notes.discrepancies[0].kind, "unique")
        self.assertEqual(
            notes.discrepancies[0].positions, {"BBC News": "named him as Ryder Williams"}
        )

    def test_differences_ordered_by_kind_and_capped(self):
        def unique(outlet, text):
            return {"kind": "unique", "topic": text,
                    "positions": [{"outlet": outlet, "says": text}]}
        notes = self.check({"differences": [
            unique("NPR", "detail one"),
            {"kind": "figures", "topic": "altitude drop", "positions": [
                {"outlet": "BBC News", "says": "17,000 feet"},
                {"outlet": "NPR", "says": "14,000 feet"},
            ]},
            unique("France 24", "detail two"),
            {"kind": "conflict", "topic": "took office", "positions": [
                {"outlet": "BBC News", "says": "took office in May"},
                {"outlet": "Al Jazeera", "says": "took office in July"},
            ]},
            unique("The Guardian", "detail three"),
        ]})
        self.assertEqual(
            [d.kind for d in notes.discrepancies],
            ["conflict", "figures", "unique", "unique"],
        )

    def test_one_sided_or_unknown_outlet_is_dropped(self):
        notes = self.check({"discrepancies": [{"topic": "toll", "positions": [
            {"outlet": "BBC News", "says": "13 dead"},
            {"outlet": "Made Up Times", "says": "40 dead"},
        ]}]})
        self.assertEqual(notes.discrepancies, [])


class RenderingTest(unittest.TestCase):

    def test_document_order_and_sources_line(self):
        story = StoryBrief("Bridge", ["NPR", "BBC News"], [], "## Bridge\n\nText.")
        digest = StoryBrief("Economic & Monetary Policy", ["ECB"], [],
                            "## Economic & Monetary Policy\n\nRates held.")
        doc = RenderingStage().render(
            "Overview.", [story], digest, now=datetime(2026, 9, 30, 9, 5)
        )
        self.assertTrue(doc.startswith("# Daily Brief: 9:05 AM, September 30, 2026"))
        self.assertLess(doc.index("Overview."), doc.index("## Bridge"))
        self.assertLess(doc.index("## Bridge"), doc.index("## Economic"))
        self.assertTrue(doc.rstrip().endswith("Sources: BBC News, ECB, NPR."))

    def test_no_digest_means_no_economics_section(self):
        story = StoryBrief("Bridge", ["NPR"], [], "## Bridge\n\nText.")
        doc = RenderingStage().render(None, [story], None)
        self.assertNotIn("Economic", doc)


if __name__ == "__main__":
    unittest.main()


class OutputFilenameTest(unittest.TestCase):

    def test_named_by_date_as_mon_day_year(self):
        self.assertEqual(default_output_filename(datetime(2026, 10, 3)), "OCT_03_2026.md")

    def test_two_digit_day_is_kept(self):
        self.assertEqual(default_output_filename(datetime(2026, 12, 25)), "DEC_25_2026.md")


class TruncateWordsTest(unittest.TestCase):

    def test_short_text_is_left_alone(self):
        self.assertEqual(_truncate_words("one two\nthree", 5), "one two\nthree")

    def test_long_text_is_cut_and_marked(self):
        self.assertEqual(_truncate_words("a b c d e", 3), "a b c […]")
