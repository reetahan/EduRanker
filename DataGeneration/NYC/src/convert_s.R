# AIM:
# s to -1 and s^ to -2 


convert_col <- function(x) {
  case_when(
    grepl("^[0-9,]+$", x) ~ as.numeric(gsub(",", "", x)),
    x == "s"              ~ -1,
    x == "s^"             ~ -2,
    TRUE                  ~ NA_real_
  )
}
