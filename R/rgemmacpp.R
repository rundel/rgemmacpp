#' @export gemma
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
    list(...)
  )

  new(gemma_interface, args)
}
