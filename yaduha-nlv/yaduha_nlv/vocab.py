"""Orizaba Nahuatl (nlv) vocabulary.

Target forms are given in the Orizaba Nahuatl practical orthography used by
INALI / ILV / local materials: <k> for /k/, <u> for /u/ and semivowel /w/,
<j> for /h/, <ku> for /kw/, single <l> (not <ll>). The absolutive suffix
/-tl ~ -tli ~ -li ~ -in/ is written on citation forms.

All target forms trace to at least one of the following sources:

  (1) Gold target captions in americasnlp2026 dev/pilot splits for nlv
      (nlv_001 … nlv_051). Tokens like `siuamej`, `tlakamej`, `tokatl`,
      `kouatl`, `xochitl`, `senpoalxochitl`, `ojtli`, `tepetl`, `tepeyo`,
      `mixtli`, `ateskatl`, `ameyali`, `kuauitl`, `kuajyo`, `mila`,
      `kali`/`kaltsintli`, `karro`, `kamion`, `foto`, `xonbelo`, `kamisa`,
      `pantalon`, `tlakotl`, `kuitlapan`, `mai`/`ima`, `tlakentli`, `tlali`,
      `tetl`, `nakatl`, `chili`, `xonakatl`, `tsopelikatl`, `tsilanko`,
      `chilmoli`, `tlaxkali`, `tlakuali`, `xiuitl`, `yolotl`, `xonokuauitl`,
      `xonokuilmej`, `olochkuauitl`, `pitsauaxochitl`, `mantajkaltsintli`,
      `ixtlauak`, `tlapoualistli`, `tlajtolkopa`, `kiniatl`, `kuautitlamitl`,
      `mamayotsintli`, `tlalkamojtli`, `ojtenpa`, `inakastlan`, `itlanpa`,
      `ijkuak`, `kanin`, `pampa`, `uan`, `tlen`, `non`, `nin`, `nochtin`,
      and verbs `tejteki`, `chiua`, `pia`, `ilia`, `yaualoa`,
      `toka`, `kueponi`, `nemi`, `chanti`, `uitsi`, `uetsi`, `miki`,
      `tekipanoa`, `tlejko`, `nesi`, `uaki`, `kochi`, `tlachia`,
      `tlapouia`, `tlakentia`, `kixtia`, `paleuia`, `motlalia`, `tepanoa`,
      `chia`, `tlakoloua`, `pejua`, `nextia`, `kisa`, `pitsuaki`, `mokopa`,
      `tetliokolia` all appear directly or in inflected forms in these
      captions.
  (2) Andrews, J. R. (2003). *Introduction to Classical Nahuatl* (Revised).
      University of Oklahoma Press. — stem shapes and Classical cognates.
  (3) Launey, M. (2011). *An Introduction to Classical Nahuatl*.
      Cambridge University Press. — subject/object prefix paradigms,
      absolutive suffix, preterit o-…-k morphology.
  (4) Tuggy, D. (1979). *Tetelcingo Nahuatl*. In R. Langacker (ed.),
      Studies in Uto-Aztecan Grammar. SIL. — closely related dialect;
      source for stems `nejnemi`, `motlalia`, `itskuintli`, `kauayo`,
      `miston`, `tototl`, `atoyatl`, `iluikatl`, `istatl`.
  (5) SIL-Mexico online Orizaba Nahuatl materials
      (mexico.sil.org/resources/archives/17693) — confirms practical
      orthography and the diminutive/reverential `-tsin ~ -tsintli`;
      attests color term `texotik` (blue), `nextik` (gray), and the
      productive adjectival suffix `-tik`. Also the pattern of Spanish
      loans (`karro`, `kamion`, `foto`, `sombrero`, `lentes`, `kafen`).
"""
from yaduha.language import VocabEntry

NOUNS = [
    # ---- People ----
    VocabEntry(english="woman",        target="siuatl"),         # (1) siuamej, siuatsintli
    VocabEntry(english="man",          target="tlakatl"),        # (1) tlakamej, tlakatsin
    VocabEntry(english="person",       target="maseuali"),       # (1) maseualmej
    VocabEntry(english="people",       target="maseualmej"),     # (1)
    VocabEntry(english="child",        target="konetl"),         # (2,3)
    VocabEntry(english="boy",          target="topiltsi"),       # (1) topiltsi, topilmej
    VocabEntry(english="young_man",    target="telpochtli"),     # (1) telpochtli, telpochmej
    VocabEntry(english="young_woman",  target="ichpochtli"),     # (1) ichpochtli, ichpoka
    VocabEntry(english="mother",       target="tenantsin"),      # (1)
    VocabEntry(english="father",       target="tajtli"),         # (2,3)
    VocabEntry(english="grandmother",  target="tenantsin"),      # (1) polite/elder form
    VocabEntry(english="grandfather",  target="tajtli"),         # (2,3)
    VocabEntry(english="friend",       target="ikniu"),          # (3)
    VocabEntry(english="townspeople",  target="altepetlakamej"), # (1)
    VocabEntry(english="villagers",    target="altepetlakamej"), # (1) synonym of townspeople
    VocabEntry(english="speaker",      target="nauatekatl"),     # (1) nauatekaj
    VocabEntry(english="group",        target="sentilistli"),    # (2,3) "gathering"
    VocabEntry(english="crowd",        target="sentilistli"),    # (2,3) same stem
    # ---- Animals ----
    VocabEntry(english="dog",          target="itskuintli"),     # (4)
    VocabEntry(english="cat",          target="miston"),         # (4)
    VocabEntry(english="horse",        target="kauayo"),         # (4)
    VocabEntry(english="cow",          target="kuakue"),         # (5)
    VocabEntry(english="chicken",      target="piyo"),           # (5)
    VocabEntry(english="bird",         target="tototl"),         # (4)
    VocabEntry(english="fish",         target="michin"),         # (2,3)
    VocabEntry(english="snake",        target="kouatl"),         # (1) kouatl
    VocabEntry(english="spider",       target="tokatl"),         # (1) tokatl, tokatsin
    VocabEntry(english="worm",         target="okuili"),         # (1) xonokuilmej
    VocabEntry(english="insect",       target="yolkatl"),        # (2,3)
    VocabEntry(english="mantis",       target="mantis"),         # (1) loan, nlv_043
    # ---- Plants & food ----
    VocabEntry(english="tree",         target="kuauitl"),        # (1) kuauitl, xonokuauitl, olochkuauitl
    VocabEntry(english="branch",       target="kuaumaitl"),      # (2,3) kuauh+mai "tree-hand"
    VocabEntry(english="forest",       target="kuajyo"),         # (1) kuajyo
    VocabEntry(english="woods",        target="kuautitlamitl"),  # (1) kuautitlamitl
    VocabEntry(english="plant",        target="xiuitl"),         # (1) xiuitl
    VocabEntry(english="flower",       target="xochitl"),        # (1) xochitl + compounds
    VocabEntry(english="marigold",     target="senpoalxochitl"), # (1)
    VocabEntry(english="banana",       target="kiniatl"),        # (1) kiniaxochitl, kiniaxiuitl
    VocabEntry(english="leaf",         target="iswatl"),         # (2,3)
    VocabEntry(english="grass",        target="sakatl"),         # (2,3)
    VocabEntry(english="corn",         target="sintli"),         # (2,3)
    VocabEntry(english="cornfield",    target="mila"),           # (1) mila, ouamila
    VocabEntry(english="food",         target="tlakuali"),       # (1) tlakuali
    VocabEntry(english="tortilla",     target="tlaxkali"),       # (1) tlaxkali
    VocabEntry(english="bread",        target="tlaxkali"),       # (1) same stem used for bread
    VocabEntry(english="meat",         target="nakatl"),         # (1) nakatl
    VocabEntry(english="chile",        target="chili"),          # (1) chilmej
    VocabEntry(english="onion",        target="xonakatl"),       # (1) xonakatl
    VocabEntry(english="sugar",        target="tsopelikatl"),    # (1) tsopelikatl
    VocabEntry(english="cilantro",     target="tsilanko"),       # (1) tsilanko
    VocabEntry(english="mole",         target="chilmoli"),       # (1) chilmoli
    VocabEntry(english="fruit",        target="xokotl"),         # (2,3)
    VocabEntry(english="seed",         target="achtli"),         # (2,3)
    # ---- Nature & places ----
    VocabEntry(english="water",        target="atl"),            # (2,3); cf. ameyali, ateskatl in (1)
    VocabEntry(english="fire",         target="tletl"),          # (2,3)
    VocabEntry(english="sun",          target="tonati"),         # (2,3,4)
    VocabEntry(english="moon",         target="metstli"),        # (2,3)
    VocabEntry(english="star",         target="sitlalin"),       # (2,3)
    VocabEntry(english="sky",          target="iluikatl"),       # (4)
    VocabEntry(english="cloud",        target="mixtli"),         # (1) mixtli, tlamixtentok
    VocabEntry(english="fog",          target="ayautli"),        # (2,3)
    VocabEntry(english="earth",        target="tlali"),          # (1) tlali
    VocabEntry(english="ground",       target="tlali"),          # (1) tlali
    VocabEntry(english="stone",        target="tetl"),           # (1) tetl, temej
    VocabEntry(english="mountain",     target="tepetl"),         # (1) tepetl
    VocabEntry(english="hill",         target="tepetl"),         # (1) tepetl / tepetsintli
    VocabEntry(english="mountain_range", target="tepeyo"),       # (1) tepeyo
    VocabEntry(english="river",        target="atoyatl"),        # (4)
    VocabEntry(english="lake",         target="ateskatl"),       # (1) ateskatl
    VocabEntry(english="pond",         target="ateskatl"),       # (1) small body of still water
    VocabEntry(english="stream",       target="ueyatsintli"),    # (1) ueyatsintli
    VocabEntry(english="spring",       target="ameyali"),        # (1) ameyali
    VocabEntry(english="road",         target="ojtli"),          # (1) ojtli
    VocabEntry(english="path",         target="ojtli"),          # (1) ojtli
    VocabEntry(english="roadside",     target="ojtenpa"),        # (1) ojtenpa
    VocabEntry(english="sidewalk",     target="ojtenpa"),        # (1) same lexeme — edge of the road
    VocabEntry(english="plain",        target="ixtlauak"),       # (1) ixtlauak
    VocabEntry(english="field",        target="ixtlauak"),       # (1) open country
    VocabEntry(english="wind",         target="ejekatl"),        # (2,3)
    VocabEntry(english="rain",         target="kiauitl"),        # (2,3)
    VocabEntry(english="salt",         target="istatl"),         # (4)
    VocabEntry(english="background",   target="ikampa"),         # (2,3) "behind, in back of"
    # ---- Built environment & artifacts ----
    VocabEntry(english="house",        target="kali"),           # (1) kali, kaltsintli, kalmej
    VocabEntry(english="building",     target="kali"),           # (1) same stem
    VocabEntry(english="town",         target="altepetl"),       # (1) altepetlakamej, ialtepetlaken
    VocabEntry(english="village",      target="altepetl"),       # (1)
    VocabEntry(english="chapel",       target="mantajkaltsintli"), # (1)
    VocabEntry(english="church",       target="mantajkaltsintli"), # (1)
    VocabEntry(english="market",       target="tianki"),         # (2,3)
    VocabEntry(english="plaza",        target="tianki"),         # (2,3) open gathering space
    VocabEntry(english="fence",        target="tlatsakuili"),    # (1) itlatsakuil
    VocabEntry(english="wall",         target="tepamitl"),       # (2,3)
    VocabEntry(english="door",         target="kaltentli"),      # (1) kaltentli
    VocabEntry(english="roof",         target="kaltsonteko"),    # (2,3) kal+tsonteko "house-head"
    VocabEntry(english="floor",        target="tlalpan"),        # (2,3) "on the ground"
    VocabEntry(english="table",        target="tlapechtli"),     # (1) tlapechtli
    VocabEntry(english="platform",     target="tlapechtli"),     # (2,3) tlapechtli = raised surface
    VocabEntry(english="podium",       target="tlapechtli"),     # (2,3) same stem
    VocabEntry(english="bench",        target="ikpali"),         # (2,3)
    VocabEntry(english="chair",        target="ikpali"),         # (2,3)
    VocabEntry(english="stick",        target="tlakotl"),        # (1) tlakotl
    VocabEntry(english="knife",        target="tepostli"),       # (2,3)
    VocabEntry(english="pot",          target="komitl"),         # (2,3)
    VocabEntry(english="plate",        target="kaxitl"),         # (2,3)
    VocabEntry(english="pan",          target="komitl"),         # (2,3) reuse
    VocabEntry(english="coffin",       target="mikkakaxa"),      # (2,3) mik-ka (dead) + kaxa (Sp. box)
    VocabEntry(english="cemetery",     target="mikkaltsintli"),  # (2,3) mik-kal "dead-house"
    VocabEntry(english="car",          target="karro"),          # (1) karro — Spanish loan in data
    VocabEntry(english="truck",        target="karro"),          # (1)
    VocabEntry(english="bus",          target="kamion"),         # (1) kamion
    VocabEntry(english="photo",        target="foto"),           # (1) foto
    VocabEntry(english="picture",      target="foto"),           # (1)
    VocabEntry(english="image",        target="foto"),           # (1)
    VocabEntry(english="camera",       target="foto"),           # (1) camera = photo-device; reuse loan
    VocabEntry(english="paper",        target="amatl"),          # (2,3)
    VocabEntry(english="book",         target="amoxtli"),        # (2,3)
    VocabEntry(english="clothing",     target="tlaken"),         # (1) tlaken, xochitlakemitl
    VocabEntry(english="traditional_clothing", target="altepetlaken"), # (1) ialtepetlaken
    VocabEntry(english="shirt",        target="kamisa"),         # (1) kamisa
    VocabEntry(english="jacket",       target="tlaken"),         # (1) generic outer garment
    VocabEntry(english="coat",         target="tlaken"),         # (1) generic outer garment
    VocabEntry(english="vest",         target="tlaken"),         # (1) generic body garment
    VocabEntry(english="pants",        target="pantalon"),       # (1) pantalon
    VocabEntry(english="shawl",        target="xonbelo"),        # (1) xonbelo (rebozo)
    VocabEntry(english="scarf",        target="xonbelo"),        # (1) same garment type
    VocabEntry(english="hat",          target="sombrero"),       # (5) Spanish loan, standard in Orizaba
    VocabEntry(english="glasses",      target="lentes"),         # (5) Spanish loan, standard
    VocabEntry(english="sunglasses",   target="lentes"),         # (5) same lexeme
    VocabEntry(english="bag",          target="koxtali"),        # (2,3,4)
    VocabEntry(english="mask",         target="xayakatl"),       # (2,3)
    # ---- Abstract / speech ----
    VocabEntry(english="language",     target="tlajtoli"),       # (1) intlajtolkopa
    VocabEntry(english="story",        target="tlapoualistli"),  # (1) tlapoualistli
    VocabEntry(english="word",         target="tlajtoli"),       # (2,3)
    VocabEntry(english="color",        target="tlapali"),        # (2,3)
    VocabEntry(english="name",         target="tokaitl"),        # (2,3)
    # ---- Body parts & relational ----
    VocabEntry(english="hand",         target="mai"),            # (1) ima (his hand)
    VocabEntry(english="back",         target="kuitlapan"),      # (1) ikuitlapan
    VocabEntry(english="head",         target="tsontekon"),      # (2,3)
    VocabEntry(english="face",         target="ixtli"),          # (2,3)
    VocabEntry(english="eye",          target="ixtli"),          # (2,3)
    VocabEntry(english="foot",         target="ikxitl"),         # (2,3)
    VocabEntry(english="leg",          target="metstli"),        # (2,3)
    VocabEntry(english="heart",        target="yolotl"),         # (1) ken yolotl
    VocabEntry(english="beard",        target="tentsontli"),     # (2,3) "lip-hair"
    VocabEntry(english="hair",         target="tsontli"),        # (2,3)
    VocabEntry(english="mouth",        target="kamaktli"),       # (2,3)
    VocabEntry(english="ear",          target="nakastli"),       # (2,3)
]

TRANSITIVE_VERBS = [
    VocabEntry(english="eat",      target="kua"),        # (1) tikkuaj
    VocabEntry(english="drink",    target="i"),          # (2,3)
    VocabEntry(english="see",      target="itta"),       # (2,3)
    VocabEntry(english="look_at",  target="tlachia"),    # (1) tlachixtok
    VocabEntry(english="hear",     target="kaki"),       # (1) kikajtokej
    VocabEntry(english="listen_to", target="kaki"),      # (1) same stem
    VocabEntry(english="make",     target="chiua"),      # (1) kichiua
    VocabEntry(english="do",       target="chiua"),      # (1) same stem
    VocabEntry(english="cut",      target="tejteki"),    # (1) kitejteki
    VocabEntry(english="carry",    target="uika"),       # (2,3)
    VocabEntry(english="bring",    target="ualkui"),     # (2,3)
    VocabEntry(english="take",     target="kui"),        # (2,3)
    VocabEntry(english="buy",      target="koua"),       # (2,3)
    VocabEntry(english="sell",     target="namaka"),     # (2,3)
    VocabEntry(english="give",     target="maka"),       # (2,3); cf. tetliokolia (1)
    VocabEntry(english="offer",    target="tetliokolia"), # (1) kintetliokolilia
    VocabEntry(english="wash",     target="paka"),       # (2,3); cf. tlapajpakaj (1)
    VocabEntry(english="cook",     target="ixka"),       # (2,3)
    VocabEntry(english="plant",    target="toka"),       # (2,3); cf. kitookaskej in (1)
    VocabEntry(english="hold",     target="pia"),        # (1) kipixtok
    VocabEntry(english="have",     target="pia"),        # (1) same stem — "possess" sense
    VocabEntry(english="wear",     target="pia"),        # (1) used for garment possession/wearing
    VocabEntry(english="contain",  target="pia"),        # (1) same stem
    VocabEntry(english="call",     target="ilia"),       # (1) kiluiaj, kiliaj
    VocabEntry(english="say",      target="ilia"),       # (1) same stem
    VocabEntry(english="tell",     target="ilia"),       # (1) same stem
    VocabEntry(english="read",     target="tlapoua"),    # (2,3) tlapoua = count/read
    VocabEntry(english="count",    target="tlapoua"),    # (2,3) same stem
    VocabEntry(english="surround", target="yaualoa"),    # (1) kiyaualojtok
    VocabEntry(english="help",     target="paleuia"),    # (1) mopaleuiaj, tlapaleuia
    VocabEntry(english="wait_for", target="chia"),       # (1) kichixtok
    VocabEntry(english="wall_off", target="tepanoa"),    # (1) okintepankej
    VocabEntry(english="take_out", target="kixtia"),     # (1) kimokixtiliaj
    VocabEntry(english="bury",     target="toka"),       # (1) kitookaskej
    VocabEntry(english="love",     target="tlasojtla"),  # (2,3)
    VocabEntry(english="teach",    target="machtia"),    # (2,3)
    VocabEntry(english="know",     target="ixmati"),     # (2,3)
    VocabEntry(english="gather",   target="sentilia"),   # (2,3)
    VocabEntry(english="find",     target="asi"),        # (2,3)
    VocabEntry(english="watch",    target="tlachia"),    # (1) tlachixtok (is watching)
    VocabEntry(english="clean",    target="tlachipaua"), # (1) tlachipajtok (is clean)
    VocabEntry(english="show",     target="nextia"),     # (1) kinextia
    VocabEntry(english="hang",     target="piloa"),      # (2,3)
    VocabEntry(english="cover",    target="tlapachoa"),  # (1) tlamixtentok ~ covered in clouds
    VocabEntry(english="cross",    target="panoa"),      # (2,3)
    VocabEntry(english="begin",    target="pejua"),      # (1) yopejki
    VocabEntry(english="finish",   target="tlami"),      # (2,3)
]

INTRANSITIVE_VERBS = [
    VocabEntry(english="sleep",    target="kochi"),      # (2,3)
    VocabEntry(english="walk",     target="nemi"),       # (1) nemi; cf. nejnemi (4)
    VocabEntry(english="run",      target="totoka"),     # (1) totoka
    VocabEntry(english="go",       target="yaui"),       # (2,3); cf. tiuaij (1)
    VocabEntry(english="come",     target="uitsi"),      # (1) uitsej
    VocabEntry(english="arrive",   target="asi"),        # (2,3)
    VocabEntry(english="climb",    target="tlejko"),     # (1) tlejkoskej
    VocabEntry(english="fall",     target="uetsi"),      # (1) uetsi
    VocabEntry(english="sit",      target="motlalia"),   # (4)
    VocabEntry(english="stand",    target="ikak"),       # (2,3)
    VocabEntry(english="rest",     target="moseuia"),    # (1) moseuijtok
    VocabEntry(english="live",     target="chanti"),     # (1) chantij
    VocabEntry(english="die",      target="miki"),       # (1) omomikili
    VocabEntry(english="work",     target="tekipanoa"),  # (1) otekipanoto
    VocabEntry(english="play",     target="auiltia"),    # (2,3)
    VocabEntry(english="laugh",    target="uetska"),     # (2,3)
    VocabEntry(english="smile",    target="uetska"),     # (2,3) same stem
    VocabEntry(english="cry",      target="choka"),      # (2,3)
    VocabEntry(english="dance",    target="mijtotia"),   # (2,3)
    VocabEntry(english="sing",     target="kuika"),      # (2,3)
    VocabEntry(english="fly",      target="patlani"),    # (2,3)
    VocabEntry(english="swim",     target="aneloa"),     # (2,3)
    VocabEntry(english="bathe",    target="maltia"),     # (2,3)
    VocabEntry(english="bloom",    target="kueponi"),    # (1) kueponij
    VocabEntry(english="appear",   target="nesi"),       # (1) nesi
    VocabEntry(english="dry_up",   target="uaki"),       # (1) youakik, yopitsuakik
    VocabEntry(english="speak",    target="tlajtoa"),    # (2,3); cf. intlajtolkopa
    VocabEntry(english="talk",     target="tlajtoa"),    # (2,3) same stem
    VocabEntry(english="converse", target="motlapouia"), # (1) motlapouiaj
    VocabEntry(english="chat",     target="motlapouia"), # (1) same stem
    VocabEntry(english="dress",    target="motlakentia"),# (1) motlakentok
    VocabEntry(english="leave",    target="kisa"),       # (1) okiski
    VocabEntry(english="exit",     target="kisa"),       # (1) same stem
    VocabEntry(english="return",   target="mokuepa"),    # (1) mokopa
    VocabEntry(english="curve",    target="tlakoloua"),  # (1) tlakoloua
    VocabEntry(english="wind",     target="tlakoloua"),  # (1) same stem — "(of road) to wind"
    VocabEntry(english="flow",     target="motoyaua"),   # (2,3)
    VocabEntry(english="shine",    target="tlanesi"),    # (2,3)
    VocabEntry(english="be_located", target="ka"),       # (1) katej (3pl), kajki (sg)
    VocabEntry(english="exist",    target="ka"),         # (1) same stem
    VocabEntry(english="be_cloudy", target="tlamixtemi"), # (1) tlamixtentok
]
