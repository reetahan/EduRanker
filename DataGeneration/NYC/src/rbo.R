# RBO for top-k lists


rbo_topk <- function(L1, L2, p = 0.9, k = NULL) {
  if (is.null(k)) k <- max(length(L1), length(L2))
  k <- min(k, max(length(L1), length(L2)))
  
  s1 <- character(0); s2 <- character(0)
  sum_term <- 0
  
  for (d in 1:k) {
    if (d <= length(L1)) s1 <- c(s1, L1[d])
    if (d <= length(L2)) s2 <- c(s2, L2[d])
    
    overlap_d <- length(intersect(s1, s2))
    A_d <- overlap_d / d
    
    sum_term <- sum_term + (p^(d - 1)) * A_d
  }
  
  # finite extrapolated RBO for top-k lists
  # (common practical form; stable for short lists like k=12)
  rbo_ext <- (1 - p) * sum_term + (p^k) * (length(intersect(L1, L2)) / k)
  rbo_ext
}
