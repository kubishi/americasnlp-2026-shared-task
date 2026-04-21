"""Yucatec Maya (yua) vocabulary.

Sources and citations for every target form:

* TRAIN = 50 training captions in americasnlp2026/data/{dev,pilot}/yua.jsonl,
  inspected directly. Every noun/verb target form marked TRAIN is attested
  verbatim (or as a near-orthographic variant) in at least one of those
  captions. Specific caption ids are noted where useful.
* BOHN = Bohnemeyer, J. (2002), *The Grammar of Time Reference in Yucatec
  Maya* (LINCOM).
* LEH = Lehmann, C., *Yucatec Maya grammar* notes (christianlehmann.eu).
* SIL = SIL Mexico Yucatec Maya resources (www.sil.org/resources/.../yua).
* BRICK = Bricker, Po'ot Yah & Dzul de Po'ot (1998), *A Dictionary of the
  Maya Language As Spoken in Hocabá, Yucatán*.
* ALMG/INALI = Maayáaj orthography references (Instituto Nacional de
  Lenguas Indígenas, Mexico).

Orthography: ALMG/INALI-style modern Maayáaj t'aan spelling. Apostrophe
marks glottalisation (k', p', t', ch', ts', vowel-glottalisation V'),
doubled vowels are long (aa, ee, ii, oo, uu), acute accent marks high
tone on long vowels (áa, ée, íi, óo, úu).
"""
from yaduha.language import VocabEntry


NOUNS = [
    # -- People (animate) -------------------------------------------------
    VocabEntry(english="person",      target="máak"),        # TRAIN: 003, 012, 022, 026, 028, 043 ; BRICK
    VocabEntry(english="man",         target="winik"),       # TRAIN: 030, 032, 037 ; BRICK
    VocabEntry(english="woman",       target="ch'uup"),      # TRAIN: 050 ; BRICK
    VocabEntry(english="lady",        target="ko'olel"),     # BRICK, SIL
    VocabEntry(english="child",       target="paal"),        # BRICK, SIL
    VocabEntry(english="elder",       target="nojoch máak"), # TRAIN: 022 (phrasal)
    VocabEntry(english="people",      target="wíinik"),      # TRAIN: 035, 038 (wíiniko'ob)
    VocabEntry(english="boy",         target="xi'ipal"),     # BRICK
    VocabEntry(english="girl",        target="xch'úupal"),   # BRICK

    # -- Animals (animate) ------------------------------------------------
    VocabEntry(english="cat",         target="miis"),        # TRAIN: 001, 027, 042, 044 ; BRICK
    VocabEntry(english="dog",         target="peek'"),       # TRAIN: 038, 039 ; BRICK
    VocabEntry(english="chicken",     target="kaax"),        # TRAIN: 018, 028, 049 ; BRICK
    VocabEntry(english="rooster",     target="t'eel"),       # BRICK
    VocabEntry(english="bird",        target="ch'íich'"),    # TRAIN: 034 ; BRICK
    VocabEntry(english="butterfly",   target="péepen"),      # TRAIN: 048 ; BRICK
    VocabEntry(english="fish",        target="kay"),         # BRICK
    VocabEntry(english="ant",         target="síinik"),      # TRAIN: 020 (sínik) ; BRICK
    VocabEntry(english="horse",       target="tsíimin"),     # BRICK
    VocabEntry(english="cow",         target="wakax"),       # SIL
    VocabEntry(english="pig",         target="k'éek'en"),    # BRICK
    VocabEntry(english="deer",        target="kéej"),        # BRICK
    VocabEntry(english="snake",       target="kaan"),        # TRAIN: 032 ; BRICK
    VocabEntry(english="animal",      target="ba'alche'"),   # TRAIN: 033 (ba'alche'ob) ; BRICK

    # -- Plants & trees (plant class) ------------------------------------
    VocabEntry(english="tree",        target="che'"),        # TRAIN: passim ; BRICK
    VocabEntry(english="flower",      target="lool"),        # TRAIN: 010, 013, 015, 017, 051 ; BRICK
    VocabEntry(english="leaf",        target="le'"),         # TRAIN: 016, 020, 046 ; BRICK
    VocabEntry(english="fruit",       target="ich"),         # TRAIN: 016, 046, 047 ; BRICK
    VocabEntry(english="branch",      target="k'ab"),        # TRAIN: 047, 048 ; BRICK
    VocabEntry(english="root",        target="mootz"),       # BRICK
    VocabEntry(english="forest",      target="k'áax"),       # TRAIN: 013, 017, 021, 030, 031 ; BRICK
    VocabEntry(english="plant",       target="pak'al"),      # TRAIN: 008, 022, 029 ; BRICK
    VocabEntry(english="wall_plant",  target="pak'"),        # TRAIN: 012, 019 ; BRICK
    VocabEntry(english="seed",        target="íinaj"),       # BRICK
    VocabEntry(english="corn_plant",  target="xiim"),        # TRAIN: 029, 032, 033 ; BRICK
    VocabEntry(english="corn",        target="ixi'im"),      # BRICK, SIL
    VocabEntry(english="papaya",      target="puut"),        # TRAIN: 016 ; BRICK
    VocabEntry(english="sapodilla",   target="ya'"),         # TRAIN: 020 ; BRICK
    VocabEntry(english="banana",      target="ja'as"),       # TRAIN: 016 ; BRICK
    VocabEntry(english="palm",        target="xa'an"),       # TRAIN: 008, 045, 046 ; BRICK
    VocabEntry(english="bean",        target="bu'ul"),       # TRAIN: 029 ; BRICK
    VocabEntry(english="white_bean",  target="iib"),         # TRAIN: 043 ; BRICK
    VocabEntry(english="squash",      target="k'uum"),       # TRAIN: 031 ; BRICK
    VocabEntry(english="pumpkin",     target="k'uum"),       # TRAIN: 031 ; BRICK
    VocabEntry(english="vine",        target="aak'"),        # TRAIN: 037 ; BRICK
    VocabEntry(english="firewood",    target="si'"),         # TRAIN: 037 ; BRICK
    VocabEntry(english="thorn",       target="k'i'ix"),      # TRAIN: 034 (k'i'ixil) ; BRICK

    # -- Built environment / places (inanimate) --------------------------
    VocabEntry(english="house",       target="naj"),         # TRAIN: passim ; BRICK
    VocabEntry(english="big_house",   target="noj naj"),     # TRAIN: 035 ; BRICK
    VocabEntry(english="door",        target="jool"),        # TRAIN: 038 ; BRICK
    VocabEntry(english="doorway",     target="joolnaj"),     # TRAIN: 006 ; BRICK
    VocabEntry(english="wall",        target="pak'il"),      # TRAIN: 044 ; BRICK
    VocabEntry(english="fence",       target="koot"),        # TRAIN: 029, 039 ; BRICK
    VocabEntry(english="road",        target="bej"),         # TRAIN: 051 ; BRICK
    VocabEntry(english="highway",     target="carretera"),   # TRAIN: 039 (Spanish loan)
    VocabEntry(english="bridge",      target="k'áat bej"),   # BRICK
    VocabEntry(english="town",        target="kaaj"),        # BRICK
    VocabEntry(english="garden",      target="solar"),       # TRAIN: 029 (Spanish loan)
    VocabEntry(english="school",      target="najilxook"),   # TRAIN: 007 ; BRICK
    VocabEntry(english="bench",       target="k'áanche'"),   # TRAIN: 019 ; BRICK
    VocabEntry(english="table",       target="mesa"),        # TRAIN: 026 (Spanish loan)

    # -- Natural environment (inanimate) ---------------------------------
    VocabEntry(english="water",       target="ja'"),         # TRAIN: 006 ; BRICK
    VocabEntry(english="fire",        target="k'áak'"),      # TRAIN: 024, 050 ; BRICK
    VocabEntry(english="smoke",       target="buts'"),       # TRAIN: 005 ; BRICK
    VocabEntry(english="earth",       target="lu'um"),       # TRAIN: 009, 018, 031 ; BRICK
    VocabEntry(english="ground",      target="lu'um"),       # TRAIN: 009, 018 ; BRICK
    VocabEntry(english="stone",       target="tunich"),      # TRAIN: 031 ; BRICK
    VocabEntry(english="sun",         target="k'íin"),       # TRAIN: 027, 034 ; BRICK
    VocabEntry(english="day",         target="k'íin"),       # TRAIN: 027, 034 ; BRICK
    VocabEntry(english="moon",        target="uj"),          # BRICK
    VocabEntry(english="light",       target="sáasil"),      # TRAIN: 007, 027, 034 ; BRICK
    VocabEntry(english="cloud",       target="muunyal"),     # TRAIN: 032, 033 ; BRICK
    VocabEntry(english="sky",         target="ka'an"),       # BRICK
    VocabEntry(english="star",        target="éek'"),        # TRAIN: 017 ; BRICK
    VocabEntry(english="rain",        target="cháak"),       # BRICK
    VocabEntry(english="hay",         target="su'uk"),       # TRAIN: 005, 012 ; BRICK
    VocabEntry(english="grass",       target="su'uk"),       # TRAIN: 005, 012 ; BRICK
    VocabEntry(english="nest",        target="k'u'"),        # TRAIN: 034 ; BRICK
    VocabEntry(english="shadow",      target="oochel"),      # TRAIN: 012 (yoochel) ; BRICK

    # -- Artifacts / objects (inanimate) ---------------------------------
    VocabEntry(english="book",        target="áanalte'"),    # TRAIN: 002 ; BRICK
    VocabEntry(english="language",    target="t'aan"),       # TRAIN: 002 ; BRICK
    VocabEntry(english="bottle",      target="botella"),     # TRAIN: 006 (loan)
    VocabEntry(english="computer",    target="computadora"), # TRAIN: 003 (loan)
    VocabEntry(english="container",   target="nu'ukul"),     # TRAIN: 004, 026 ; BRICK
    VocabEntry(english="pot",         target="nu'ukul"),     # TRAIN: 004, 026 ; BRICK
    VocabEntry(english="bag",         target="bolsa"),       # TRAIN: 026 (loan)
    VocabEntry(english="bucket",      target="cubeta"),      # TRAIN: 042 (loan)
    VocabEntry(english="truck",       target="kisbuuts'"),   # TRAIN: 014, 045 ; BRICK
    VocabEntry(english="car",         target="kisbuuts'"),   # TRAIN: 014, 045 ; BRICK
    VocabEntry(english="bicycle",     target="bicicleta"),   # TRAIN: 040 (loan)
    VocabEntry(english="hat",         target="p'óok"),       # BRICK
    VocabEntry(english="cloth",       target="nok'"),        # TRAIN: 035 ; BRICK
    VocabEntry(english="clothing",    target="nok'"),        # TRAIN: 035 ; BRICK
    VocabEntry(english="huipil",      target="huipil"),      # TRAIN: 022
    VocabEntry(english="flag",        target="bandera"),     # TRAIN: 035 (banderrillas; loan)
    VocabEntry(english="sheet",       target="lámina"),      # TRAIN: 042 (loan)
    VocabEntry(english="switch",      target="apagador"),    # TRAIN: 041 (loan)
    VocabEntry(english="color",       target="boonil"),      # TRAIN: 012, 022, 035 ; BRICK
    VocabEntry(english="drawing",     target="boon"),        # TRAIN: 050 ; BRICK

    # -- Food / consumables (inanimate) ----------------------------------
    VocabEntry(english="tortilla",    target="waaj"),        # TRAIN: 025, 036 ; BRICK
    VocabEntry(english="tamale",      target="tamal"),       # TRAIN: 028 (tamali')
    VocabEntry(english="food",        target="janal"),       # BRICK
    VocabEntry(english="meal",        target="janal"),       # BRICK
    VocabEntry(english="honey",       target="kaab"),        # TRAIN: 005 ; BRICK
    VocabEntry(english="chocolate",   target="chukwa'"),     # TRAIN: 019 ; BRICK
    VocabEntry(english="coffee",      target="café"),        # TRAIN: 004 (loan)
    VocabEntry(english="nixtamal",    target="nixtamal"),    # TRAIN: 024 (Spanish loan)
    VocabEntry(english="drink",       target="uk'ul"),       # TRAIN: 004 ; BRICK
]


TRANSITIVE_VERBS = [
    # Stored in the "incompletive / imperfective" form (stem + -ik or
    # -tik), which the imperfective aspect particle ``ku`` selects.
    # All entries cross-checked against BRICK unless otherwise noted.
    VocabEntry(english="cook",     target="tsiktik"),    # TRAIN: 028 ; BRICK (tsik-)
    VocabEntry(english="make",     target="beetik"),     # BRICK (beet-)
    VocabEntry(english="pat",      target="patik"),      # TRAIN: 028 ; BRICK
    VocabEntry(english="shell",    target="xiixtik"),    # TRAIN: 043 ; BRICK (xiix-)
    VocabEntry(english="tie",      target="k'aaxik"),    # TRAIN: 026, 037 ; BRICK (k'ax-)
    VocabEntry(english="bind",     target="k'aaxik"),    # TRAIN: 026, 037 ; BRICK
    VocabEntry(english="shine",    target="jultik"),     # TRAIN: 034 ; BRICK (jul-)
    VocabEntry(english="scratch",  target="tóochik"),    # TRAIN: 018 ; BRICK
    VocabEntry(english="eat",      target="jaantik"),    # BRICK (jaan-)
    VocabEntry(english="drink",    target="uk'ik"),      # BRICK (uk'-)
    VocabEntry(english="see",      target="ilik"),       # BRICK (il-)
    VocabEntry(english="look_at",  target="wilik"),      # TRAIN: 027, 032 ; BRICK
    VocabEntry(english="know",     target="k'ajóoltik"), # BRICK
    VocabEntry(english="hear",     target="u'uyik"),     # BRICK
    VocabEntry(english="say",      target="a'alik"),     # BRICK
    VocabEntry(english="carry",    target="kuchik"),     # BRICK
    VocabEntry(english="bring",    target="taasik"),     # BRICK
    VocabEntry(english="plant",    target="pak'ik"),     # BRICK (pak'-)
    VocabEntry(english="wash",     target="p'o'ik"),     # BRICK (p'o'-)
    VocabEntry(english="buy",      target="manik"),      # BRICK (man-)
    VocabEntry(english="find",     target="kaxtik"),     # BRICK
    VocabEntry(english="take",     target="ch'a'ik"),    # BRICK (ch'a'-)
    VocabEntry(english="take_out", target="jo'sik"),     # TRAIN: 005 (jo'saaj PFV) ; BRICK
    VocabEntry(english="put",      target="ts'áaik"),    # TRAIN: 026 ; BRICK
    VocabEntry(english="read",     target="xokik"),      # BRICK (xok-)
    VocabEntry(english="write",    target="ts'íibtik"),  # BRICK
    VocabEntry(english="open",     target="je'ik"),      # BRICK
    VocabEntry(english="close",    target="k'alik"),     # BRICK
    VocabEntry(english="hold",     target="mach'ik"),    # BRICK
    VocabEntry(english="cut",      target="xotik"),      # BRICK
    VocabEntry(english="hit",      target="loxik"),      # BRICK
    VocabEntry(english="love",     target="yaakuntik"),  # BRICK
    VocabEntry(english="want",     target="k'áatik"),    # BRICK
    VocabEntry(english="pick_up",  target="lúubsik"),    # BRICK
    VocabEntry(english="wear",     target="búukintik"),  # BRICK
    VocabEntry(english="build",    target="jets'ik"),    # BRICK
]


INTRANSITIVE_VERBS = [
    # Stored in "incompletive / imperfective" form (typically stem +
    # -Vl status suffix). Stative / positional verbs are stored in
    # their -Vkbal stative form, which is invariant and predicative
    # on its own (no aspect marker). All cross-checked against BRICK.
    VocabEntry(english="sleep",    target="weenel"),     # TRAIN: 001, 041 ; BRICK (ween-)
    VocabEntry(english="walk",     target="xíimbal"),    # TRAIN: 030 ; BRICK
    VocabEntry(english="pass",     target="máan"),       # TRAIN: 042 ; BRICK
    VocabEntry(english="come",     target="taal"),       # BRICK
    VocabEntry(english="go",       target="bin"),        # BRICK
    VocabEntry(english="arrive",   target="k'uchul"),    # BRICK
    VocabEntry(english="run",      target="áalkab"),     # BRICK
    VocabEntry(english="fly",      target="xik'nal"),    # BRICK
    VocabEntry(english="live",     target="kuxtal"),     # BRICK
    VocabEntry(english="die",      target="kíimil"),     # TRAIN: 005 (kíimen) ; BRICK
    VocabEntry(english="rest",     target="je'lel"),     # BRICK
    VocabEntry(english="work",     target="meyaj"),      # BRICK
    VocabEntry(english="play",     target="báaxal"),     # BRICK
    VocabEntry(english="bloom",    target="loolankil"),  # TRAIN: 008 (loolankií) ; BRICK
    VocabEntry(english="grow",     target="ch'íijil"),   # BRICK
    VocabEntry(english="sing",     target="k'aay"),      # BRICK
    VocabEntry(english="dance",    target="óok'ot"),     # BRICK
    VocabEntry(english="cry",      target="ok'ol"),      # BRICK
    VocabEntry(english="laugh",    target="che'ej"),     # BRICK
    VocabEntry(english="fall",     target="lúubul"),     # BRICK
    VocabEntry(english="bathe",    target="ichkíil"),    # BRICK
    # Positional / stative -Vkbal forms (used as main predicates)
    VocabEntry(english="sit",      target="kulukbal"),   # TRAIN: 038 ; BRICK
    VocabEntry(english="stand",    target="wa'akbal"),   # TRAIN: 008, 022, 032 ; BRICK
    VocabEntry(english="lie",      target="chilikbal"),  # TRAIN: 039, 044 ; BRICK
    VocabEntry(english="hang",     target="ch'uyukbal"), # TRAIN: 034, 035, 042 ; BRICK
    VocabEntry(english="crouch",   target="p'ukukbal"),  # TRAIN: 027 ; BRICK
    VocabEntry(english="gather",   target="much'ukbal"), # TRAIN: 035 ; BRICK
]
