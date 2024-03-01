#' @export
gemma_args = function() {
  df = arg_help()
  df$help = stringr::str_squish(df$help)

  df
}


#' @export
gemma = function(tokenizer, compressed_weights, model, ...) {

  stopifnot(file.exists(tokenizer))
  stopifnot(file.exists(compressed_weights))

  tokenizer = normalizePath(tokenizer)
  compressed_weights = normalizePath(compressed_weights)

  args = c(
    list(
      tokenizer=tokenizer,
      compressed_weights=compressed_weights,
      model=model
    ),
    lapply(
      list(...), as.character
    )
  )

  methods::new(gemma_interface, args)
}
