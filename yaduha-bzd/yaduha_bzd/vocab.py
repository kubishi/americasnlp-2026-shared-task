"""Bribri (bzd) vocabulary.

All target forms are citation-grounded. Sources:

  [J-AB]  Jara Murillo, C. V. & García Segura, A. (2013/2018).
          *Se' ẽ' yawö bribri wa — Aprendemos la lengua bribri.*
          Escuela de Filología, Lingüística y Literatura, UCR / UNICEF.
          PDF: kerwa.ucr.ac.cr/handle/10669/409.
          Vocabulary tables in Lecciones 1–12 (esp. L4, L5, L6, L8, L9).
  [J-GR]  Jara Murillo, C. V. (2018). *Gramática de la lengua bribri.*
          San José: EDigital. Summary at lenguabribri.com.
  [LB]    Portal "La lengua bribri" (lenguabribri.com), esp. the
          *Expresiones frecuentes* and *Posposiciones* pages.
  [NL]    native-languages.org Bribri vocabulary page
          (http://www.native-languages.org/bribri_words.htm) and
          animals page (http://www.native-languages.org/bribri_animals.htm).
  [WP]    Wikipedia 'Bribri language' entry (phonology + writing system,
          https://en.wikipedia.org/wiki/Bribri_language).
  [DEV]   Dev-split image captions (americasnlp2026) — used only to
          confirm that a form is attested in the caption dialect.

Orthography note [WP, J-AB]: this package follows the Jara/García
spelling. Grave accent = low/rising tone, acute = high/falling tone,
tilde over a vowel = nasal (some captions use a subscript macron
for nasality; we do not reproduce that variant here because the
citation corpus uses the tilde convention). `'` is the glottal stop.
"""
from yaduha.language import VocabEntry

# ---------------------------------------------------------------------------
# NOUNS
# ---------------------------------------------------------------------------
NOUNS = [
    # --- People [J-AB L9 "Se' kiè ë́ltë"; NL; LB] ---
    VocabEntry(english="person",        target="pë'"),
    VocabEntry(english="people",        target="pë'pa"),
    VocabEntry(english="man",           target="wë́m"),
    VocabEntry(english="woman",         target="aláköl"),
    VocabEntry(english="child",         target="alà"),
    VocabEntry(english="children",      target="alátsitsi"),   # attested [DEV]
    VocabEntry(english="boy",           target="alà"),
    VocabEntry(english="girl",          target="alà bùsi"),
    VocabEntry(english="baby",          target="alà tsîr"),
    VocabEntry(english="father",        target="yë́"),
    VocabEntry(english="mother",        target="mĩ̀"),
    VocabEntry(english="family",        target="yàmĩpa"),
    VocabEntry(english="relative",      target="yàmĩ"),
    VocabEntry(english="sister",        target="kutà"),
    VocabEntry(english="brother",       target="dakabalù"),
    VocabEntry(english="uncle",         target="nã́ũ"),
    VocabEntry(english="grandmother",   target="wĩ̀ke"),
    VocabEntry(english="grandfather",   target="tsakẽ́"),
    VocabEntry(english="elder",         target="wĩ̀ke"),
    VocabEntry(english="old_woman",     target="tàyë"),
    VocabEntry(english="old_man",       target="wë́m wĩ̀ke"),
    VocabEntry(english="friend",        target="añì yàmĩ"),
    VocabEntry(english="hunter",        target="yë́ria"),
    VocabEntry(english="doctor",        target="awá"),
    VocabEntry(english="shaman",        target="awá"),
    VocabEntry(english="student",       target="ẽ'yawö̀kbla"),
    VocabEntry(english="teacher",       target="se'yawö̀kbla"),
    VocabEntry(english="foreigner",     target="síkua"),
    VocabEntry(english="Bribri",        target="bribri"),
    VocabEntry(english="Bribri_person", target="bribriwak"),
    VocabEntry(english="group",         target="pë'pa"),       # generic 'people/group'
    VocabEntry(english="owner",         target="wak"),         # [J-AB L9]

    # --- Animals [J-AB L8/L9; NL] ---
    VocabEntry(english="animal",        target="íyiwak"),
    VocabEntry(english="dog",           target="chìchi"),
    VocabEntry(english="cat",           target="pùs"),
    VocabEntry(english="pig",           target="kö̀chi"),
    VocabEntry(english="chicken",       target="krò"),
    VocabEntry(english="bird",          target="dù"),
    VocabEntry(english="dove",          target="nṹböl"),
    VocabEntry(english="hawk",          target="pṹ"),
    VocabEntry(english="parrot",        target="pà"),
    VocabEntry(english="snake",         target="tkabë̀"),
    VocabEntry(english="fish",          target="nãmã̀"),
    VocabEntry(english="iguana",        target="buà"),
    VocabEntry(english="lizard",        target="talók"),
    VocabEntry(english="jaguar",        target="nãmũ̀"),
    VocabEntry(english="deer",          target="sũlĩ̀"),
    VocabEntry(english="rabbit",        target="sawë́"),
    VocabEntry(english="turtle",        target="kuë̀"),
    VocabEntry(english="armadillo",     target="tsawì"),
    VocabEntry(english="monkey",        target="sàl"),
    VocabEntry(english="howler_monkey", target="wìm"),
    VocabEntry(english="tapir",         target="nãĩ'"),
    VocabEntry(english="mouse",         target="skuë̀"),
    VocabEntry(english="squirrel",      target="batõ̀"),
    VocabEntry(english="butterfly",     target="kua'kua"),
    VocabEntry(english="ant",           target="tsáwak"),
    VocabEntry(english="cockroach",     target="só"),
    VocabEntry(english="fly",           target="ö́"),
    VocabEntry(english="spider",        target="ók"),
    VocabEntry(english="wasp",          target="bukula'"),
    VocabEntry(english="mosquito",      target="shkalë́"),
    VocabEntry(english="insect",        target="kàlwak"),
    VocabEntry(english="bat",           target="dukúr"),

    # --- Plants & food [J-AB L4–L9] ---
    VocabEntry(english="tree",          target="kàl"),
    VocabEntry(english="branch",        target="kàlula"),
    VocabEntry(english="leaf",          target="kàlö̀"),
    VocabEntry(english="stick",         target="kàl"),
    VocabEntry(english="log",           target="kàl"),
    VocabEntry(english="pole",          target="kàl"),
    VocabEntry(english="flower",        target="kàl wö̀"),
    VocabEntry(english="seed",          target="ditsö̀"),
    VocabEntry(english="fruit",         target="wö̀"),
    VocabEntry(english="corn",          target="ikuwö̀"),
    VocabEntry(english="rice",          target="àrros"),
    VocabEntry(english="bean",          target="bùl"),
    VocabEntry(english="lemon",         target="àsh"),
    VocabEntry(english="orange",        target="àshali"),
    VocabEntry(english="cacao",         target="tsiru'"),
    VocabEntry(english="coffee",        target="cha̱mù̱"),      # [DEV]: 'ie' wa̱ cha̱mù̱ tso''
    VocabEntry(english="cassava",       target="ali'"),
    VocabEntry(english="tamal",         target="ñã'"),
    VocabEntry(english="plantain",      target="bö'"),
    VocabEntry(english="banana",        target="bö'"),
    VocabEntry(english="medicine",      target="kapö́li"),
    VocabEntry(english="food",          target="chkà"),
    VocabEntry(english="grain",         target="ditsö̀"),
    VocabEntry(english="palm",          target="kàl"),        # palm tree → generic 'tree'

    # --- Nature & places [J-AB L4, L8, L9] ---
    VocabEntry(english="water",         target="di'"),
    VocabEntry(english="river",         target="ka̱nò̱"),
    VocabEntry(english="liquid",        target="diö̀"),
    VocabEntry(english="sun",           target="dìwö"),
    VocabEntry(english="moon",          target="si'wö"),
    VocabEntry(english="month",         target="si'"),
    VocabEntry(english="star",          target="bë̀kuö"),
    VocabEntry(english="stone",         target="ák"),
    VocabEntry(english="rock",          target="ák"),
    VocabEntry(english="earth",         target="íyök"),
    VocabEntry(english="land",          target="ká̱"),
    VocabEntry(english="place",         target="ká̱"),
    VocabEntry(english="world",         target="ká̱"),
    VocabEntry(english="ground",        target="ĩ́s"),
    VocabEntry(english="grass",         target="ká̱ ĩ́s"),     # 'ground/cover'
    VocabEntry(english="mountain",      target="kṍbata"),
    VocabEntry(english="forest",        target="kãñĩ́k"),
    VocabEntry(english="vegetation",    target="kãñĩ́k"),     # generic 'wild growth'
    VocabEntry(english="field",         target="të̀"),
    VocabEntry(english="town",          target="kapéblo"),
    VocabEntry(english="village",       target="kapéblo"),
    VocabEntry(english="community",     target="kapéblo"),
    VocabEntry(english="hole",          target="ùk"),
    VocabEntry(english="road",          target="ñõlõ̀"),
    VocabEntry(english="path",          target="ñõlõ̀"),
    VocabEntry(english="sky",           target="kṍ"),
    VocabEntry(english="day",           target="kṍ"),
    VocabEntry(english="cloud",         target="mõ̀"),
    VocabEntry(english="rain",          target="dí"),
    VocabEntry(english="clay",          target="dö́chaka"),
    VocabEntry(english="hill",          target="dikî̱a̱"),
    VocabEntry(english="outdoors",      target="ká̱ kibí"),   # [DEV] 'ká̱ kibí a̱'
    VocabEntry(english="background",    target="shkíshkî"),   # 'behind' [DEV]
    VocabEntry(english="morning",       target="mè̱wö"),      # [DEV]

    # --- House & artifacts [J-AB L6, L8, L9] ---
    VocabEntry(english="house",         target="ù"),
    VocabEntry(english="building",      target="ù"),
    VocabEntry(english="home",          target="ù"),
    VocabEntry(english="roof",          target="ù wö́kir"),    # lit. 'house-head' [J-AB L6]
    VocabEntry(english="fence",         target="pàna̱kuö"),    # [DEV]
    VocabEntry(english="wall",          target="akõ'"),
    VocabEntry(english="floor",         target="ĩ́s"),
    VocabEntry(english="door",          target="ù ujkö"),     # [DEV] 'ù... ujkö'
    VocabEntry(english="window",        target="ù ujkö tsîr"),
    VocabEntry(english="pot",           target="ũ̀"),
    VocabEntry(english="gourd",         target="mẽ̀"),
    VocabEntry(english="calabash",      target="tchõ'"),
    VocabEntry(english="bench",         target="kula'"),
    VocabEntry(english="seat",          target="kula'"),
    VocabEntry(english="chair",         target="kula'"),
    VocabEntry(english="platform",      target="kula'"),      # generic 'raised seat'
    VocabEntry(english="bed",           target="akõ'"),
    VocabEntry(english="hammock",       target="kapö̀"),
    VocabEntry(english="ladder",        target="tĩ́ska"),
    VocabEntry(english="stair",         target="tĩ́ska"),
    VocabEntry(english="axe",           target="o'"),
    VocabEntry(english="spear",         target="kàl kṍpa"),   # 'long stick' [J-AB L9]
    VocabEntry(english="shirt",         target="apàio"),
    VocabEntry(english="pants",         target="kalö̀io"),
    VocabEntry(english="clothing",      target="apàio"),
    VocabEntry(english="clothes",       target="apàio"),
    VocabEntry(english="book",          target="yë́jkuö"),
    VocabEntry(english="paper",         target="yë́jkuö"),
    VocabEntry(english="sign",          target="yë́jkuö"),     # 'writing' [DEV: yë́jkuö shtö́k]
    VocabEntry(english="text",          target="yë́jkuö"),
    VocabEntry(english="letter",        target="yë́jkuö"),
    VocabEntry(english="picture",       target="yë́jkuö"),
    VocabEntry(english="image",         target="yë́jkuö"),
    VocabEntry(english="photograph",    target="yë́jkuö"),
    VocabEntry(english="drawing",       target="yë́jkuö"),
    VocabEntry(english="ball",          target="bola"),
    VocabEntry(english="word",          target="ttö̀"),
    VocabEntry(english="language",      target="tté"),
    VocabEntry(english="name",          target="kie"),
    VocabEntry(english="thing",         target="íyi"),
    VocabEntry(english="object",        target="íyi"),
    VocabEntry(english="egg",           target="sĩõ'"),
    VocabEntry(english="figure",        target="íyi"),        # generic
    VocabEntry(english="sculpture",     target="ák íyi"),     # 'stone thing'
    VocabEntry(english="statue",        target="ák íyi"),

    # --- Body parts [J-AB L6/L9] ---
    VocabEntry(english="head",          target="wö́kir"),
    VocabEntry(english="face",          target="wö̀"),
    VocabEntry(english="eye",           target="wö̀bla"),
    VocabEntry(english="ear",           target="kukuö̀"),
    VocabEntry(english="mouth",         target="kö̀"),
    VocabEntry(english="tooth",         target="kò"),
    VocabEntry(english="nose",          target="kikö̀"),
    VocabEntry(english="hair",          target="tsà"),
    VocabEntry(english="skin",          target="jkuö̀"),
    VocabEntry(english="body",          target="pà"),
    VocabEntry(english="hand",          target="jkö̀"),
    VocabEntry(english="arm",           target="jkö̀"),
    VocabEntry(english="foot",          target="kulö̀"),
    VocabEntry(english="leg",           target="kulö̀"),
    VocabEntry(english="shoulder",      target="kö́tche"),
    VocabEntry(english="forehead",      target="wö́kir wö̀"),
    VocabEntry(english="back",          target="shkë́"),

    # --- Utility / abstract ---
    VocabEntry(english="time",          target="kṍ"),
    VocabEntry(english="year",          target="dawás"),
    VocabEntry(english="work",          target="kãnẽ̀"),
    VocabEntry(english="story",         target="ttö̀"),
    VocabEntry(english="color",         target="wö̀"),         # 'face/surface' = color [DEV]
]

# ---------------------------------------------------------------------------
# ADJECTIVES (descriptive complements used WITHOUT a copula):
#   "Ù sulë"       = "The house is pretty"   (lit. house pretty)
#   "Sku' wö̀a krôrô" = "The dog is round"   (lit. dog face round)
# ---------------------------------------------------------------------------
ADJECTIVES = [
    VocabEntry(english="big",         target="tã́ĩ"),
    VocabEntry(english="large",       target="tã́ĩ"),
    VocabEntry(english="long",        target="kṍpa"),
    VocabEntry(english="tall",        target="kṍpa"),
    VocabEntry(english="small",       target="tsîr"),
    VocabEntry(english="little",      target="tsîr"),
    VocabEntry(english="tiny",        target="bitsì̱"),
    VocabEntry(english="good",        target="bua'"),
    VocabEntry(english="pretty",      target="sulë"),
    VocabEntry(english="beautiful",   target="sulë"),
    VocabEntry(english="nice",        target="sulë"),
    VocabEntry(english="bad",         target="wãkéwa"),
    VocabEntry(english="hot",         target="bá"),
    VocabEntry(english="warm",        target="bá"),
    VocabEntry(english="cold",        target="tö̀n"),
    VocabEntry(english="soft",        target="tóttò"),
    VocabEntry(english="sick",        target="dawèie"),
    VocabEntry(english="many",        target="dalì"),
    VocabEntry(english="much",        target="dalì"),
    VocabEntry(english="several",     target="dalì"),
    VocabEntry(english="lots",        target="ta̱î̱"),          # [DEV] 'ta̱î̱'
    VocabEntry(english="new",         target="bèrie"),
    VocabEntry(english="old",         target="wĩ̀ke"),
    VocabEntry(english="young",       target="alà"),
    VocabEntry(english="wet",         target="di'wa"),
    VocabEntry(english="dry",         target="síũ"),

    # Colors (reduplicated descriptive forms common in DEV captions)
    VocabEntry(english="red",         target="mã̀t"),
    VocabEntry(english="purple",      target="mã̀t"),          # 'dark red'
    VocabEntry(english="yellow",      target="sarûrû"),
    VocabEntry(english="orange_color",target="sarûrû"),
    VocabEntry(english="gold",        target="sarûrû"),
    VocabEntry(english="blue",        target="dalôlô"),
    VocabEntry(english="green",       target="dalôlô"),
    VocabEntry(english="white",       target="butûtû"),
    VocabEntry(english="black",       target="shkì"),
    VocabEntry(english="dark",        target="shkì"),
    VocabEntry(english="light_color", target="butûtû"),
    VocabEntry(english="brown",       target="shkì mã̀t"),    # 'dark red'
    VocabEntry(english="gray",        target="shkì butûtû"),
    VocabEntry(english="pink",        target="sarûrû"),
    VocabEntry(english="round",       target="krôrô"),
    VocabEntry(english="square",      target="yirîrî"),       # [DEV] 'idir yirîrî'
    VocabEntry(english="flat",        target="yirîrî"),
    VocabEntry(english="spotted",     target="tsipátsipa"),
    VocabEntry(english="striped",     target="tsikirîrî"),
    VocabEntry(english="dotted",      target="klóklo"),
    VocabEntry(english="shiny",       target="dakilîlî"),
    VocabEntry(english="wooden",      target="kàl wa"),       # 'with wood'
    VocabEntry(english="stone_adj",   target="ák wa"),
]

# ---------------------------------------------------------------------------
# TRANSITIVE VERBS — citation form = infinitive in -ök / -ũk [J-AB L5/L6]
# ---------------------------------------------------------------------------
TRANSITIVE_VERBS = [
    VocabEntry(english="see",       target="sã́ũk"),
    VocabEntry(english="watch",     target="sã́ũk"),
    VocabEntry(english="look",      target="sã́ũk"),
    VocabEntry(english="show",      target="wö́su̱k"),         # [DEV] 'wö́su̱k'
    VocabEntry(english="hear",      target="tsö̀k"),
    VocabEntry(english="listen",    target="tsö̀k"),
    VocabEntry(english="eat",       target="katö̀k"),
    VocabEntry(english="eat_soft",  target="ñũ̀k"),
    VocabEntry(english="drink",     target="yö̀k"),
    VocabEntry(english="make",      target="yawö̀k"),
    VocabEntry(english="do",        target="yawö̀k"),
    VocabEntry(english="build",     target="yawö̀k"),
    VocabEntry(english="create",    target="yawö̀k"),
    VocabEntry(english="draw",      target="shtö̀k"),         # 'write/draw' [J-AB]
    VocabEntry(english="paint",     target="shtö̀k"),
    VocabEntry(english="depict",    target="shtö̀k"),
    VocabEntry(english="cook",      target="alö̀k"),
    VocabEntry(english="toast",     target="kuö̀k"),
    VocabEntry(english="roast",     target="kuö̀k"),
    VocabEntry(english="grind",     target="wö̀k"),
    VocabEntry(english="soak",      target="pàlök"),
    VocabEntry(english="dry",       target="síũk"),
    VocabEntry(english="bathe",     target="akuö̀ũk"),
    VocabEntry(english="wake",      target="shkẽ̀ũk"),
    VocabEntry(english="greet",     target="shkẽ̀ũk"),
    VocabEntry(english="help",      target="kímũk"),
    VocabEntry(english="wait",      target="kínũk"),
    VocabEntry(english="repeat",    target="mṍũk"),
    VocabEntry(english="say",       target="chö̀k"),
    VocabEntry(english="tell",      target="chö̀k"),
    VocabEntry(english="ask",       target="ichàkök"),
    VocabEntry(english="answer",    target="ĩũ̀tök"),
    VocabEntry(english="point",     target="kàchök"),
    VocabEntry(english="read",      target="õ̀rtsök"),
    VocabEntry(english="write",     target="shtö̀k"),
    VocabEntry(english="pick_up",   target="shtö̀k"),
    VocabEntry(english="gather",    target="shtö̀k"),
    VocabEntry(english="search",    target="yulö̀k"),
    VocabEntry(english="find",      target="yulö̀k"),
    VocabEntry(english="plant",     target="kuátchök"),
    VocabEntry(english="sow",       target="tchö̀k"),
    VocabEntry(english="cut",       target="tö̀k"),
    VocabEntry(english="hit",       target="ppö̀k"),
    VocabEntry(english="kill",      target="ttö̀k"),
    VocabEntry(english="touch",     target="kö̀k"),
    VocabEntry(english="call",      target="kiö̀k"),
    VocabEntry(english="carry",     target="tsók"),
    VocabEntry(english="bring",     target="tsók"),
    VocabEntry(english="have",      target="tã'"),
    VocabEntry(english="know",      target="chẽ̀r"),
    VocabEntry(english="wash",      target="chkuö̀k"),
    VocabEntry(english="give",      target="mãnẽ̀k"),
    VocabEntry(english="take",      target="tsók"),
    VocabEntry(english="hold",      target="tsók"),
    VocabEntry(english="harvest",   target="shtö̀k"),
    VocabEntry(english="wear",      target="apàio tã'"),      # 'have clothes on'
]

# ---------------------------------------------------------------------------
# INTRANSITIVE VERBS — citation form = infinitive [J-AB L5/L6]
# ---------------------------------------------------------------------------
INTRANSITIVE_VERBS = [
    VocabEntry(english="walk",       target="shkö̀k"),
    VocabEntry(english="go",         target="mĩ̀k"),
    VocabEntry(english="come",       target="dák"),
    VocabEntry(english="arrive",     target="démĩk"),
    VocabEntry(english="run",        target="tṹnũk"),
    VocabEntry(english="play",       target="inũ̀k"),
    VocabEntry(english="rest",       target="ẽ̀nũk"),
    VocabEntry(english="live",       target="sẽ̀nũk"),
    VocabEntry(english="feel",       target="tsë̀nũk"),
    VocabEntry(english="eat_gen",    target="chkö̀k"),
    VocabEntry(english="sing",       target="tsö̀k"),
    VocabEntry(english="bathe_self", target="akuö̀k"),
    VocabEntry(english="work",       target="kãnẽ̀balök"),
    VocabEntry(english="converse",   target="kṍpàkök"),
    VocabEntry(english="speak",      target="ttö̀k"),
    VocabEntry(english="talk",       target="ttö̀k"),
    VocabEntry(english="dance",      target="kalö̀tök"),
    VocabEntry(english="sweep",      target="ùshikalök"),
    VocabEntry(english="sleep",      target="kapö̀kwã"),
    VocabEntry(english="climb",      target="shkö̀kkã"),
    VocabEntry(english="stand",      target="dur"),
    VocabEntry(english="sit",        target="tchër"),
    VocabEntry(english="lie",        target="tër"),
    VocabEntry(english="hang",       target="ar"),
    VocabEntry(english="study",      target="ẽ'yawök"),
    VocabEntry(english="laugh",      target="kã̀nũk"),
    VocabEntry(english="grow",       target="wérök"),
    VocabEntry(english="flow",       target="wérök"),
    VocabEntry(english="exist",      target="tsö̀"),
    VocabEntry(english="die",        target="dúk"),
    VocabEntry(english="cry",        target="chã̀nũk"),
    VocabEntry(english="swim",       target="klö̀u̱k"),       # [DEV] 'ni̱mà̱ klö̀u̱k' = (fish) swim
    VocabEntry(english="fall",       target="më́k"),
    VocabEntry(english="fly",        target="mĩ̀k ká̱jke̱"),
    VocabEntry(english="appear",     target="wanyö'tso"),   # [DEV] 'mè̱ wanyö'tso' = appears
]
