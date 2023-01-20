"""
Constants.
"""

# `|` == \n == place of hyphenation

# no hyphen
#  "tavalyi|ígéretei" -> "tavalyi ígéretei"
NO_HYPHEN_LABEL = "0"

# word-breaking hyphen, disappears when dehyphenated
#  "meg-|győző" -> "meggyőző"
BREAKING_HYPHEN_LABEL = "1"

# like 1, but breaks a long di- or trigraph into two short ones
#  "visz-|szaélés" -> "visszaélés"
DIGRAPH_HYPHEN_LABEL = "2"

# orthographic hyphen within a word, does not disappear when dehyphenated
#  "2-|án" -> "2-án"
ORTHOGRAPHIC_HYPHEN_LABEL = "3"

# hyphen that is followed by whitespace / "multiplication"
#  "Adó-|és Pénzügyi" -> "Adó- és Pénzügyi"
HYPHEN_PLUS_SPACE_LABEL = "4"

# hyphen that is followed by whitespace / "dash"
#  "Drobilichét -|aki" -> "Drobilichét - aki"
DASH_PLUS_SPACE_LABEL = "5"

# flawed hyph: after hyphen at the beginning of word
#  "légtérellenőrzést és -|védelmet," -> légtérellenőrzést és -védelmet,"
STARTING_HYPHEN_LABEL = "6"

# flawed hyph: between `-,`
#  "vezérigazgató -|, de" -> "vezérigazgató -, de"
DASH_PUNCT_LABEL = "7"

# flawed hyph: between `-,` within word
#  "kampány-|, majd kormányprogrammá" -> "kampány-, majd kormányprogrammá"
HYPHEN_PUNCT_LABEL = "8"


# -----

# obsolete system

# no hyphen, or hyphen that is followed by whitespace
#  a) "tavalyi|ígéretei" -> "tavalyi ígéretei"
#  b) "Adó-|és Pénzügyi" -> "Adó- és Pénzügyi" (multiplication)
#  c) "Drobilichét -|aki" -> "Drobilichét - aki" (dash)
xNO_HYPHEN_LABEL = "0"

# word-breaking hyphen, disappears when dehyphenated
#  "meg-|győző" -> "meggyőző"
xBREAKING_HYPHEN_LABEL = "1"

# like 1, but breaks a long di- or trigraph into two short ones
#  "visz-|szaélés" -> "visszaélés"
xDIGRAPH_HYPHEN_LABEL = "2"

# orthographic hyphen within a word, does not disappear when dehyphenated
#  a) "2-|án" -> "2-án"
#  b) "légtérellenőrzést és -|védelmet," -> légtérellenőrzést és -védelmet," (flawed hyph!)
#  c) "vezérigazgató -|, de" -> "vezérigazgató -, de" (flawed hyph!)
#  d) "kampány-|, majd kormányprogrammá" -> "kampány-, majd kormányprogrammá"
xORTHOGRAPHIC_HYPHEN_LABEL = "3"

# subclass of 0: b) + c)
xHYPHEN_PLUS_SPACE_LABEL = "4"

