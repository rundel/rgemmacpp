#' @title gemma argument details
#'
#' @description Returns a tibble detailing the configuration arguments for gemma class.
#'
#' @returns Returns a tibble of arguments, their default values, and help text.
#'
#' @examples
#' gemma_args()
#'
#' @export
gemma_args = function() {
  df = arg_help()
  df$help = stringr::str_squish(df$help)

  tibble::as_tibble(df)
}

#' @title gemma model
#'
#' @description Returns a gemma module which can then be prompted.
#'
#' @returns Returns a gemma module interface to an underlying C++ class.
#'
#' @examples
#' \dontrun{
#' m = gemma(
#'   tokenizer="~/Scratch/gemma/tokenizer.spm",
#'   compressed_weights="~/Scratch/gemma/2b-it-sfp.sbs",
#'   model="2b-it",
#'   multiturn="1"
#' )
#'
#' m$print_config()
#'
#' m$prompt("What are top 5 places I should visit in Durham, NC?")
#'
#' m$prompt("Which of the previous locations are best for kids?")
#'
#' m$prompts
#' m$responses
#' }
#'
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
