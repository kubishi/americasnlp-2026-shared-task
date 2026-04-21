"""Guaraní (grn) vocabulary.

Every target form is copied from one of the following sources; entries whose
form might be contested list the chosen variant in a trailing comment.

Sources
-------
- Gregores, E. & Suárez, J. (1967). *A description of colloquial Guaraní.*
  The Hague: Mouton.
- Velázquez-Castillo, M. (2002). "Grammatical relations in active systems:
  The case of Guaraní."  *Functions of Language* 9(2): 133-167.
- Guasch, A. & Ortiz, D. (1986). *Diccionario Castellano-Guaraní,
  Guaraní-Castellano*. Asunción: CEPAG.
- Wiktionary Guaraní lemma category:
  https://en.wiktionary.org/wiki/Category:Guarani_lemmas
- Dev-split captions in americasnlp2026 (grn_001 - grn_050) used to check
  that forms chosen here actually surface in attested Paraguayan Guaraní text.

Orthographic conventions
------------------------
- ẽ, ĩ, ũ, ỹ, ã, õ  : nasal vowels (standard Paraguayan Guaraní).
- '                 : glottal stop (puso).
- Acute accent marks non-default stress, following the standard orthography.
- Verb stems are listed WITHOUT the person prefix; the prefix is prepended
  by the morphology rules in `__init__.py` (a-, re-, o-, ja-, ro-, pe-, ...).
- Adjectives are uninflected and follow the head noun in an NP
  ("óga morotĩ" = white house; Gregores & Suárez 1967 §5.2).
- A few stems beginning with /'/ (glottal) surface with an epenthetic /h/
  in prefixed forms (e.g. 'u → a-h-'u "I eat"); we ignore that micro-rule
  and just concatenate the prefix with the stem, which stays readable.
"""
from yaduha.language import VocabEntry

# ---------------------------------------------------------------------------
# NOUNS
# ---------------------------------------------------------------------------
NOUNS = [
    # --- People / kinship ---
    VocabEntry(english="woman",       target="kuña"),
    VocabEntry(english="man",         target="kuimba'e"),
    VocabEntry(english="person",      target="ava"),
    VocabEntry(english="people",      target="tavayguakuéra"),
    VocabEntry(english="child",       target="mitã"),
    VocabEntry(english="boy",         target="mitãkuimba'e"),
    VocabEntry(english="girl",        target="mitãkuña"),
    VocabEntry(english="youth",       target="mitãrusu"),
    VocabEntry(english="friend",      target="angirũ"),
    VocabEntry(english="family",      target="ogayguakuéra"),
    VocabEntry(english="mother",      target="sy"),
    VocabEntry(english="father",      target="ru"),
    VocabEntry(english="hero",        target="héroe"),
    VocabEntry(english="dancer",      target="jerokyhára"),
    VocabEntry(english="worker",      target="mba'apohára"),
    VocabEntry(english="painter",     target="mba'apohára"),   # lit. "worker"; closest attested
    VocabEntry(english="builder",     target="óga apoha"),     # "house-maker"; cap. grn_005
    VocabEntry(english="singer",      target="purahéihára"),
    VocabEntry(english="pilgrim",     target="guataha"),       # cap. grn_021
    # --- Body / parts ---
    VocabEntry(english="head",        target="akã"),
    VocabEntry(english="hand",        target="po"),
    VocabEntry(english="eye",         target="resa"),
    VocabEntry(english="mouth",       target="juru"),
    VocabEntry(english="face",        target="rova"),
    VocabEntry(english="body",        target="rete"),
    VocabEntry(english="hair",        target="akãrague"),
    VocabEntry(english="shoulder",    target="ati'y"),
    VocabEntry(english="foot",        target="py"),
    VocabEntry(english="heart",       target="py'a"),
    # --- Animals ---
    VocabEntry(english="animal",      target="mymba"),
    VocabEntry(english="dog",         target="jagua"),
    VocabEntry(english="cat",         target="mbarakaja"),
    VocabEntry(english="horse",       target="kavaju"),
    VocabEntry(english="cow",         target="vaka"),
    VocabEntry(english="bull",        target="toro"),
    VocabEntry(english="pig",         target="kure"),
    VocabEntry(english="sheep",       target="ovecha"),
    VocabEntry(english="chicken",     target="ryguasu"),
    VocabEntry(english="bird",        target="guyra"),
    VocabEntry(english="fish",        target="pira"),
    VocabEntry(english="hummingbird", target="mainumby"),
    VocabEntry(english="butterfly",   target="panambi"),
    VocabEntry(english="bee",         target="eiru"),
    VocabEntry(english="snake",       target="mbói"),
    # --- Nature / places ---
    VocabEntry(english="water",       target="y"),
    VocabEntry(english="fire",        target="tata"),
    VocabEntry(english="sun",         target="kuarahy"),
    VocabEntry(english="moon",        target="jasy"),
    VocabEntry(english="star",        target="mbyja"),
    VocabEntry(english="sky",         target="yvága"),
    VocabEntry(english="earth",       target="yvy"),
    VocabEntry(english="land",        target="yvy"),
    VocabEntry(english="ground",      target="yvy"),
    VocabEntry(english="country",     target="tetã"),          # possessed: retã/ñane retã
    VocabEntry(english="town",        target="táva"),
    VocabEntry(english="city",        target="táva guasu"),
    VocabEntry(english="square",      target="pláasa"),        # public square
    VocabEntry(english="mountain",    target="yvyty"),
    VocabEntry(english="river",       target="ysyry"),
    VocabEntry(english="forest",      target="ka'aguy"),
    VocabEntry(english="road",        target="tape"),
    VocabEntry(english="path",        target="tape"),
    VocabEntry(english="stone",       target="ita"),
    VocabEntry(english="metal",       target="kuarepoti"),     # cap. grn_019 "kuarepotigui"
    VocabEntry(english="clay",        target="ñai'ũ"),
    VocabEntry(english="mud",         target="tuju"),
    VocabEntry(english="straw",       target="kapi'i"),
    VocabEntry(english="grass",       target="kapi'i"),
    VocabEntry(english="wood",        target="yvyra"),
    VocabEntry(english="tree",        target="yvyra"),
    VocabEntry(english="wicker",      target="yvyra"),         # lit. "wood/stick"; closest attested
    VocabEntry(english="flower",      target="yvoty"),
    VocabEntry(english="leaf",        target="hogue"),
    VocabEntry(english="fruit",       target="yva"),
    VocabEntry(english="seed",        target="ra'ỹi"),
    VocabEntry(english="root",        target="rapo"),          # cap. grn_030 "rapo"
    VocabEntry(english="plant",       target="ka'avo"),
    VocabEntry(english="herb",        target="ñana"),
    VocabEntry(english="medicine",    target="pohã"),
    VocabEntry(english="color",       target="sa'y"),
    VocabEntry(english="side",        target="yke"),           # cap. grn_005 "tápia yke"
    # --- Built environment ---
    VocabEntry(english="house",       target="óga"),
    VocabEntry(english="home",        target="róga"),
    VocabEntry(english="building",    target="óga"),
    VocabEntry(english="wall",        target="tápia"),         # cap. grn_005 "tápia"
    VocabEntry(english="door",        target="okẽ"),           # Wiktionary: okẽ "door"
    VocabEntry(english="window",      target="ovetã"),         # Wiktionary: ovetã "window"
    VocabEntry(english="roof",        target="ogaguy"),        # Guasch & Ortiz 1986
    VocabEntry(english="frame",       target="marco"),
    VocabEntry(english="pole",        target="yvyrakuatia"),   # lit. wooden-stick
    VocabEntry(english="flag",        target="poyvi"),         # cap. grn_001 "poyvi"
    VocabEntry(english="temple",      target="tupao"),
    VocabEntry(english="church",      target="tupao"),
    VocabEntry(english="school",      target="mbo'ehao"),
    VocabEntry(english="market",      target="mérkado"),
    VocabEntry(english="yard",        target="korapy"),
    VocabEntry(english="place",       target="tenda"),
    VocabEntry(english="table",       target="mesa"),
    VocabEntry(english="oven",        target="tatakua"),
    VocabEntry(english="bridge",      target="músu"),
    VocabEntry(english="community",   target="tekoha"),
    VocabEntry(english="brick",       target="ladríllo"),
    VocabEntry(english="background",  target="tapykue"),       # lit. back, behind
    # --- Food / drink ---
    VocabEntry(english="food",        target="tembi'u"),
    VocabEntry(english="meat",        target="so'o"),
    VocabEntry(english="corn",        target="avati"),
    VocabEntry(english="manioc",      target="mandi'o"),
    VocabEntry(english="bread",       target="mbujape"),
    VocabEntry(english="roll",        target="mbujape"),       # bread roll
    VocabEntry(english="cheese",      target="kesú"),
    VocabEntry(english="milk",        target="kamby"),
    VocabEntry(english="egg",         target="rupi'a"),
    VocabEntry(english="salt",        target="juky"),
    VocabEntry(english="sugar",       target="asuka"),
    VocabEntry(english="oil",         target="ñandy"),
    VocabEntry(english="tea",         target="ka'ay"),         # yerba mate tea
    VocabEntry(english="mate",        target="ka'ay"),
    VocabEntry(english="terere",      target="tereré"),
    VocabEntry(english="soup",        target="jukysy"),
    VocabEntry(english="chipa",       target="chipa"),
    VocabEntry(english="starch",      target="aramirõ"),
    VocabEntry(english="peanut",      target="manduvi"),
    VocabEntry(english="bean",        target="kumanda"),
    VocabEntry(english="drink",       target="mba'ehe'ẽ"),
    # --- Artifacts / clothing / objects ---
    VocabEntry(english="clothing",    target="ao"),
    VocabEntry(english="clothes",     target="ao"),
    VocabEntry(english="garment",     target="ao"),
    VocabEntry(english="hat",         target="akãngao"),
    VocabEntry(english="pot",         target="japepo"),
    VocabEntry(english="vessel",      target="kambuchi"),
    VocabEntry(english="jar",         target="kambuchi"),
    VocabEntry(english="bag",         target="vosa"),
    VocabEntry(english="basket",      target="ajaka"),         # cap. grn_022 "ajaka"
    VocabEntry(english="bucket",      target="baldéo"),        # loanword; Guasch & Ortiz 1986
    VocabEntry(english="knife",       target="kyse"),
    VocabEntry(english="fork",        target="tenedor"),       # loanword; Guasch & Ortiz 1986
    VocabEntry(english="statue",      target="ra'anga"),
    VocabEntry(english="figure",      target="ra'anga"),
    VocabEntry(english="image",       target="ra'anga"),
    VocabEntry(english="picture",     target="ta'anga"),
    VocabEntry(english="photograph",  target="ta'anga"),
    VocabEntry(english="thread",      target="inimbo"),
    VocabEntry(english="lace",        target="ñandutí"),
    VocabEntry(english="candle",      target="tataindy"),
    VocabEntry(english="book",        target="kuatiahai"),
    VocabEntry(english="paper",       target="kuatia"),
    VocabEntry(english="name",        target="téra"),
    VocabEntry(english="word",        target="ñe'ẽ"),
    VocabEntry(english="story",       target="mombe'u"),
    VocabEntry(english="dance",       target="jeroky"),
    VocabEntry(english="song",        target="purahéi"),
    VocabEntry(english="music",       target="purahéi"),
    VocabEntry(english="faith",       target="jerovia"),
    VocabEntry(english="festival",    target="arete"),
    VocabEntry(english="day",         target="ára"),
    VocabEntry(english="night",       target="pyhare"),
    VocabEntry(english="year",        target="ary"),
    VocabEntry(english="thing",       target="mba'e"),
    VocabEntry(english="work",        target="tembiapo"),
    VocabEntry(english="tool",        target="tembipuru"),     # cap. grn_005 "tembipuru-nguéra"
    VocabEntry(english="trap",        target="ñuhã"),          # cap. grn_027 "trampa" loanword, native ñuhã
]

# ---------------------------------------------------------------------------
# TRANSITIVE VERBS (stems; prefixes are added by the morphology rules)
# ---------------------------------------------------------------------------
TRANSITIVE_VERBS = [
    VocabEntry(english="eat",       target="'u"),
    VocabEntry(english="drink",     target="'u"),
    VocabEntry(english="see",       target="hecha"),
    VocabEntry(english="show",      target="hechauka"),
    VocabEntry(english="hear",      target="hendu"),
    VocabEntry(english="know",      target="kuaa"),
    VocabEntry(english="make",      target="japo"),
    VocabEntry(english="build",     target="japo"),
    VocabEntry(english="paint",     target="mosa'y"),          # lit. "make-color"
    VocabEntry(english="carry",     target="raha"),
    VocabEntry(english="take",      target="raha"),
    VocabEntry(english="hold",      target="raha"),            # closest attested
    VocabEntry(english="bring",     target="ru"),
    VocabEntry(english="cook",      target="mbojy"),
    VocabEntry(english="wash",      target="johéi"),
    VocabEntry(english="have",      target="guereko"),
    VocabEntry(english="wear",      target="guereko"),         # "have on" = "guereko"
    VocabEntry(english="find",      target="topa"),
    VocabEntry(english="use",       target="ipuru"),           # stem "puru" with i- theme vowel
    VocabEntry(english="give",      target="me'ẽ"),
    VocabEntry(english="buy",       target="jogua"),
    VocabEntry(english="sell",      target="vende"),
    VocabEntry(english="want",      target="ipota"),
    VocabEntry(english="love",      target="hayhu"),
    VocabEntry(english="help",      target="pytyvõ"),
    VocabEntry(english="tell",      target="mombe'u"),
    VocabEntry(english="say",       target="he'i"),
    VocabEntry(english="teach",     target="mbo'e"),
    VocabEntry(english="read",      target="moñe'ẽ"),
    VocabEntry(english="write",     target="haipy"),
    VocabEntry(english="plant",     target="ñotỹ"),
    VocabEntry(english="harvest",   target="monda"),           # monda "pick, harvest"
    VocabEntry(english="guard",     target="ñangareko"),
    VocabEntry(english="protect",   target="ñangareko"),
    VocabEntry(english="call",      target="henói"),
    VocabEntry(english="gather",    target="mbyaty"),
    VocabEntry(english="represent", target="hechauka"),        # "show, display"
]

# ---------------------------------------------------------------------------
# INTRANSITIVE VERBS
# ---------------------------------------------------------------------------
INTRANSITIVE_VERBS = [
    VocabEntry(english="walk",      target="guata"),
    VocabEntry(english="run",       target="ñani"),
    VocabEntry(english="sleep",     target="ke"),
    VocabEntry(english="rest",      target="pytu'u"),
    VocabEntry(english="sit",       target="guapy"),
    VocabEntry(english="stand",     target="ñembo'y"),
    VocabEntry(english="live",      target="iko"),
    VocabEntry(english="die",       target="mano"),
    VocabEntry(english="come",      target="ju"),
    VocabEntry(english="go",        target="ho"),
    VocabEntry(english="arrive",    target="guahẽ"),
    VocabEntry(english="dance",     target="jeroky"),
    VocabEntry(english="sing",      target="purahéi"),
    VocabEntry(english="work",      target="mba'apo"),
    VocabEntry(english="play",      target="ñembosarái"),
    VocabEntry(english="laugh",     target="puka"),
    VocabEntry(english="cry",       target="jahe'o"),
    VocabEntry(english="speak",     target="ñe'ẽ"),
    VocabEntry(english="fly",       target="veve"),
    VocabEntry(english="swim",      target="ytyta"),
    VocabEntry(english="fall",      target="'a"),
    VocabEntry(english="burn",      target="hendy"),
    VocabEntry(english="pray",      target="ñembo'e"),
    VocabEntry(english="gather",    target="ñembyaty"),
    VocabEntry(english="bathe",     target="jahu"),
    VocabEntry(english="hang",      target="sãingo"),          # Guasch & Ortiz 1986
]

# ---------------------------------------------------------------------------
# ADJECTIVES (uninflected; placed postnominally — "óga morotĩ" = "white house")
# ---------------------------------------------------------------------------
#
# Paraguayan Guaraní attributive adjectives are bare forms that follow the
# head noun (Gregores & Suárez 1967 §5.2; Velázquez-Castillo 2002).  A noun
# can take more than one modifier in sequence; we support a single slot on
# the Noun model to keep the schema small.
#
ADJECTIVES = [
    # --- Color ---
    VocabEntry(english="white",      target="morotĩ"),         # cap. grn_004
    VocabEntry(english="black",      target="hũ"),             # cap. grn_018 "iñakãrague hũ"
    VocabEntry(english="dark",       target="hũ"),             # reuse
    VocabEntry(english="red",        target="pytã"),           # cap. grn_004 "apeao pytã"
    VocabEntry(english="green",      target="hovy"),           # cap. grn_012 "hovy"
    VocabEntry(english="blue",       target="hovy"),
    VocabEntry(english="yellow",     target="sa'yju"),
    VocabEntry(english="brown",      target="hũngy"),          # lit. "dark-ish"; Guasch & Ortiz
    VocabEntry(english="golden",     target="sa'yju"),
    VocabEntry(english="dark-blue",  target="hovyũ"),          # cap. grn_016
    # --- Size / shape ---
    VocabEntry(english="big",        target="guasu"),
    VocabEntry(english="large",      target="guasu"),
    VocabEntry(english="small",      target="michĩ"),          # cap. grn_027 "mymba michĩva"
    VocabEntry(english="little",     target="michĩ"),
    VocabEntry(english="long",       target="puku"),           # cap. grn_004 "ijao puku"
    VocabEntry(english="tall",       target="yvate"),
    VocabEntry(english="short",      target="mbyky"),
    VocabEntry(english="round",      target="apu'a"),
    VocabEntry(english="wide",       target="pe"),
    VocabEntry(english="thick",      target="anambusu"),
    # --- Quality ---
    VocabEntry(english="good",       target="porã"),           # cap. passim
    VocabEntry(english="pretty",     target="porã"),
    VocabEntry(english="beautiful",  target="porã"),
    VocabEntry(english="bad",        target="vai"),
    VocabEntry(english="ugly",       target="vai"),
    VocabEntry(english="new",        target="pyahu"),          # cap. grn_030 "mandi'o pyahu"
    VocabEntry(english="old",        target="ymaguare"),       # cap. grn_013 "ymaguare"
    VocabEntry(english="ancient",    target="ymaguare"),
    VocabEntry(english="hot",        target="haku"),           # cap. grn_020 "haku"
    VocabEntry(english="cold",       target="ro'ysã"),         # cap. grn_002 "pohã ro'ysã"
    VocabEntry(english="sweet",      target="he'ẽ"),           # cap. grn_008 "mbujape he'ẽ"
    VocabEntry(english="bitter",     target="ro"),
    VocabEntry(english="soft",       target="vevúi"),
    VocabEntry(english="hard",       target="atã"),
    VocabEntry(english="strong",     target="mbarete"),        # cap. grn_007
    VocabEntry(english="wooden",     target="yvyragui"),       # "of wood"
    VocabEntry(english="woven",      target="pirõ"),           # Guasch & Ortiz 1986
    VocabEntry(english="sacred",     target="marangatu"),      # cap. grn_007
    VocabEntry(english="many",       target="heta"),           # cap. grn_007
]
