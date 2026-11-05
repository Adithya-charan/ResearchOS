"""A small library of five invented papers (original text, written for these tests), rendered to real PDFs. Distinct topics
make retrieval assertions meaningful; the metadata is varied on purpose so every provenance path is exercised:

    electrolytes   no title in the PDF info dict (read off the page), DOI printed on the page, creation date 2021
    sparse         year only from a 'Published 2019' line on the page
    coral          title and author declared in the PDF info dict, creation date 2020
    federated      'Published: 2022' on the page, no PDF info at all
    perovskite     creation date 2018 is the only year information
"""
from __future__ import annotations

from pathlib import Path

from tests.pdfgen import make_pdf, paginate

PAPERS: dict[str, dict] = {
    "electrolytes": {
        "title": "Solid-State Electrolytes for Lithium Batteries", "authors": "Mira Kovacs and Daniel Osei",
        "created": "D:20210412101500", "pdf_title": "", "pdf_author": "Mira Kovacs; Daniel Osei",
        "front": "DOI: 10.5555/sse.2021.014", "breaks": {"2. Methods", "4. Conclusion"},  # 3 pages
        "sections": [
            ("Abstract", [
                "Solid electrolytes could replace flammable liquid electrolytes in lithium batteries. We study a sulfide "
                "glass-ceramic whose ionic conductivity reaches twelve millisiemens per centimetre at room temperature. "
                "Symmetric cells cycled for one thousand hours without short circuits when the stack pressure stayed above "
                "five megapascals."]),
            ("1. Introduction", [
                "Lithium metal anodes promise a large gain in energy density but grow dendrites in liquid electrolytes. A "
                "dense ceramic separator can block dendrite growth in principle. In practice grain boundaries and poor "
                "interfacial contact limit the usable current density. Sulfide electrolytes are attractive because they are "
                "soft enough to be densified by cold pressing."]),
            ("2. Methods", [
                "The glass-ceramic was prepared by ball milling lithium sulfide, phosphorus pentasulfide and lithium chloride "
                "for forty hours, followed by annealing at five hundred degrees Celsius. Ionic conductivity was measured by "
                "impedance spectroscopy between blocking stainless steel electrodes from minus twenty to eighty degrees "
                "Celsius.",
                ["Pellets were pressed at three hundred megapascals and the sulfide electro-",
                 "lyte was handled in an argon glove box throughout to avoid reaction with",
                 "moisture, which would release toxic hydrogen sulfide."]]),
            ("3. Results", [
                "Conductivity followed an Arrhenius law with an activation energy of 0.28 electronvolts. Cells with a "
                "lithium metal anode cycled at one milliampere per square centimetre for one thousand hours. Short circuits "
                "appeared within one hundred hours when the stack pressure fell below two megapascals, which points to void "
                "formation at the interface."]),
            ("4. Conclusion", [
                "Sulfide glass-ceramic electrolytes reach conductivities comparable to liquid electrolytes, but stable "
                "cycling depends on applied stack pressure. Interfacial voids rather than bulk conductivity limit the "
                "current density. Future work should test thinner separators and pressure-free interlayers."]),
        ],
    },
    "sparse": {
        "title": "Sparse Attention for Long Documents", "authors": "Priya Natarajan and Tomas Berg",
        "created": None, "pdf_title": "", "pdf_author": "", "front": "Published 2019",
        "breaks": {"2. Method", "4. Discussion"},  # 3 pages
        "sections": [
            ("Abstract", [
                "Transformer language models compare every token with every other token, so attention cost grows "
                "quadratically with sequence length. We introduce a block-sparse attention pattern that lets a model read "
                "documents of sixteen thousand tokens on a single consumer GPU. Perplexity on long book passages is within "
                "two percent of dense attention."]),
            ("1. Introduction", [
                "Long documents such as legal contracts and scientific articles exceed the context window of most small "
                "language models. Truncation discards evidence that appears late in the document. Sliding window attention "
                "reduces cost but prevents distant tokens from exchanging information in a single layer."]),
            ("2. Method", [
                "Our pattern combines local blocks, a small set of global tokens and random block connections. Each query "
                "attends to at most five hundred and twelve keys regardless of the sequence length. We train with a cosine "
                "learning rate schedule, weight decay of 0.1 and gradient clipping at one."]),
            ("3. Results", [
                "On sixteen thousand token inputs the sparse model used four times less memory than dense attention and ran "
                "three times faster. Needle retrieval accuracy stayed above ninety percent up to twelve thousand tokens, then "
                "degraded when the needle was placed in the middle of the context."]),
            ("4. Discussion", [
                "Block sparsity is a hardware friendly approximation. The global tokens act as a memory that summarises "
                "distant blocks, although we did not examine what they store. We caution that perplexity does not measure "
                "whether the model uses long range evidence correctly."]),
        ],
    },
    "coral": {
        "title": "Coral Bleaching Under Marine Heatwaves", "authors": "Lena Park, Ahmed Rahimi and Sofia Duarte",
        "created": "D:20200215083000", "pdf_title": "Coral Bleaching Under Marine Heatwaves",
        "pdf_author": "Lena Park; Ahmed Rahimi; Sofia Duarte", "front": None, "breaks": {"2. Results"},  # 2 pages
        "sections": [
            ("Abstract", [
                "Coral bleaching occurs when sea surface temperature exceeds the local summer maximum for several weeks. We "
                "analysed twenty years of satellite temperature records and reef surveys from forty sites. Mortality rose "
                "sharply once accumulated heat stress passed eight degree heating weeks."]),
            ("1. Methods", [
                "Reef surveys recorded the fraction of bleached colonies along fifty metre transects at depths of five and "
                "fifteen metres. Heat stress was computed from daily satellite sea surface temperature as degree heating "
                "weeks. We fitted a logistic model of mortality against accumulated heat stress, with site as a random "
                "effect."]),
            ("2. Results", [
                "Sites with frequent past heatwaves bleached less than sites with no recent exposure, which suggests "
                "acclimatisation. Deeper transects bleached less than shallow ones, with mortality reduced by roughly a third "
                "at fifteen metres. Branching corals suffered higher mortality than massive corals under the same heat "
                "stress."]),
            ("3. Discussion", [
                "Acclimatisation may buy time but does not remove the threat. Reefs that experienced two heatwaves within "
                "three years showed no recovery of coral cover. Local measures such as reducing nutrient runoff cannot "
                "prevent bleaching, although they may improve recovery."]),
        ],
    },
    "federated": {
        "title": "Differentially Private Federated Training", "authors": "Wei Zhang and Hannah Cole",
        "created": None, "pdf_title": "", "pdf_author": "", "front": "Published: 2022", "breaks": {"2. Results"},  # 2 pages
        "sections": [
            ("Abstract", [
                "Federated learning keeps training data on user devices, but model updates can still leak information about "
                "individual examples. We combine secure aggregation with differentially private gradient clipping and noise. "
                "At a privacy budget of epsilon equal to four, accuracy on a keyboard prediction task drops by only two "
                "percentage points."]),
            ("1. Method", [
                "Each client clips its update to a fixed norm and the server adds Gaussian noise calibrated to the clipping "
                "norm after secure aggregation. A moments accountant tracks the cumulative privacy loss over training "
                "rounds. Clients are sampled uniformly at random in each round, which amplifies privacy."]),
            ("2. Results", [
                "With one thousand clients per round and a noise multiplier of 1.1 the model reached ninety two percent of "
                "the non-private accuracy after two hundred rounds. Reducing the number of clients per round below one "
                "hundred made training unstable because the noise dominated the averaged update."]),
            ("3. Limitations", [
                "Our guarantees hold for example level privacy only and assume an honest but curious server. They do not "
                "protect against clients that submit poisoned updates. The privacy accounting assumes uniform sampling, which "
                "real device availability patterns violate."]),
        ],
    },
    "perovskite": {
        "title": "Stability of Perovskite Solar Cells", "authors": "Giulia Romano and Kenji Mori",
        "created": "D:20180903120000", "pdf_title": "", "pdf_author": "Giulia Romano; Kenji Mori", "front": None,
        "breaks": {"2. Methods", "3. Results"},  # 3 pages
        "sections": [
            ("Abstract", [
                "Perovskite solar cells reach high efficiency but degrade within months in humid air. We encapsulate cells "
                "with an atomic layer deposited alumina barrier and measure efficiency over two thousand hours of damp heat. "
                "Encapsulated cells retained ninety percent of their initial efficiency."]),
            ("1. Introduction", [
                "Halide perovskites decompose when exposed to moisture, oxygen and ultraviolet light. Ion migration under "
                "bias also creates defects that reduce the open circuit voltage. Encapsulation slows moisture ingress but "
                "adds cost and can trap volatile decomposition products."]),
            ("2. Methods", [
                "Cells were fabricated on glass with a tin oxide electron transport layer and a spiro hole transport layer. "
                "The alumina barrier was deposited at eighty degrees Celsius to avoid damaging the perovskite. Damp heat "
                "testing followed the standard condition of eighty five degrees Celsius and eighty five percent relative "
                "humidity."]),
            ("3. Results", [
                "Unencapsulated reference cells lost half of their efficiency within one hundred hours. Encapsulated cells "
                "retained ninety percent after two thousand hours, though the fill factor declined steadily, indicating slow "
                "degradation at the contacts rather than in the perovskite layer."]),
        ],
    },
}


def paper_items(spec: dict) -> list[tuple]:
    items: list[tuple] = [("title", spec["title"]), ("author", spec["authors"])]
    if spec.get("front"):
        items.append(("p", spec["front"]))
    for heading, paragraphs in spec["sections"]:
        if heading in spec.get("breaks", ()):
            items.append(("pagebreak", ""))
        items.append(("h1", heading))
        for para in paragraphs:
            items.append(("lines", para) if isinstance(para, list) else ("p", para))
    return items


def write_paper(path: Path, key: str, **overrides) -> Path:
    spec = PAPERS[key]
    kwargs = dict(title=spec["pdf_title"], author=spec["pdf_author"], created=spec["created"],
                  header="Journal of Invented Results", page_numbers=True)
    kwargs.update(overrides)
    return make_pdf(path, paginate(paper_items(spec)), **kwargs)


def build_library(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    return {key: write_paper(folder / f"{key}.pdf", key) for key in PAPERS}
