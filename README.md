

<!-- README.md is generated from README.Rmd. Please edit that file -->

# rgemmacpp

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

The goal of rgemmacpp is to provide a very basic wrapper around the CLI
of Google’s [gemma.cpp](https://github.com/google/gemma.cpp).

The gemma.cpp code is included via cmake which then uses the existing
build system to compile all gemma.cpp related code as static libraries
which are then linked.

This is still very much a work in progress / proof of concept.

## Installation

You can install the development version of rgemmacpp from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("rundel/rgemmacpp")
```

In order to use the package you will need to download model weights and
tokenizer - both of these are available from [the Gemma model page on
Kaggle](https://www.kaggle.com/models/google/gemma) under
`Model Variations |> Gemma C++`. See the gemma repo’s [Quick Start -
Step
1](https://github.com/google/gemma.cpp?tab=readme-ov-file#step-1-obtain-model-weights-and-tokenizer-from-kaggle)
for more details.

Both the `2b` and `7b` models have been tested with this package -
either the `2b-it-sfp` or `7b-it-sfp` model weights are suggested as a
good place to start.

## Example

The following examples demonstrate the basics of prompting the 2b and 7b
gemma model:

``` r
library(rgemmacpp)
```

### 2b model

``` r
m = gemma(
  tokenizer="~/Scratch/gemma/tokenizer.spm",
  compressed_weights="~/Scratch/gemma/2b-it-sfp.sbs",
  model="2b-it",
  multiturn="1"
)

m$prompt("What are top 5 places I should visit in Durham, NC?")
#> 
#> 1. **Duke University Campus**
#> 2. **Durham Performing Arts Center**
#> 3. **The Duke Homestead**
#> 4. **The Duke Gardens**
#> 5. **The American Tobacco Museum**

m$prompt("Which of the previous locations are best for kids?")
#> 
#> The Duke Gardens and the American Tobacco Museum are best for kids. The Duke Gardens offer a variety of activities for kids of all ages, including a children's garden, a playground, and a petting zoo. The American Tobacco Museum offers exhibits on the history of tobacco in Durham, including a factory tour and a museum shop.

m$prompts
#> [1] "What are top 5 places I should visit in Durham, NC?"
#> [2] "Which of the previous locations are best for kids?"
m$responses
#> [1] "What are top 5 places I should visit in Durham, NC?"
#> [2] "Which of the previous locations are best for kids?"
```

### 7b model

``` r
m = gemma(
  tokenizer="~/Scratch/gemma/tokenizer.spm",
  compressed_weights="~/Scratch/gemma/7b-it-sfp.sbs",
  model="7b-it",
  multiturn="1"
)

m$prompt("What are top 5 places I should visit in Durham, NC?")
#> 
#> 1. **Duke University Campus**
#> 2. **The Museum of Life and Science**
#> 3. **The Duke Gardens**
#> 4. **The North Carolina Museum of Gems**
#> 5. **The Eno River Trail System**

m$prompt("Which of the previous locations are best for kids?")
#> 
#> The Museum of Life and Science, the Duke Gardens, and the North Carolina Museum of Gems are all great places for kids.

m$prompts
#> [1] "What are top 5 places I should visit in Durham, NC?"
#> [2] "Which of the previous locations are best for kids?"
m$responses
#> [1] "What are top 5 places I should visit in Durham, NC?"
#> [2] "Which of the previous locations are best for kids?"
```

### Configuration options

All of the currently available configuration options can be explored
through the `gemma_args()` function:

``` r
gemma_args()
#> # A tibble: 12 × 3
#>    arg                  default                help                             
#>    <chr>                <chr>                  <chr>                            
#>  1 tokenizer            ""                     Path name of tokenizer model fil…
#>  2 compressed_weights   ""                     Path name of compressed weights …
#>  3 model                ""                     Model type 2b-it (2B parameters,…
#>  4 weights              ""                     Path name of model weights (.sbs…
#>  5 max_tokens           "3072"                 Maximum number of tokens in prom…
#>  6 max_generated_tokens "2048"                 Maximum number of tokens to gene…
#>  7 temperature          "1.000000"             Temperature for top-K            
#>  8 deterministic        "0"                    Make top-k sampling deterministic
#>  9 multiturn            "0"                    Multiturn mode (if 0, this clear…
#> 10 verbosity            "1"                    Show verbose developer informati…
#> 11 num_threads          "18446744073709551615" Number of threads to use. Defaul…
#> 12 eot_line             ""                     End of turn line. When you speci…
```

Once a model has been constructed the configuration can be viewed with

``` r
m = gemma(
  tokenizer="~/Scratch/gemma/tokenizer.spm",
  compressed_weights="~/Scratch/gemma/2b-it-sfp.sbs",
  model="2b-it",
  multiturn="1"
)

m$print_config()
#> tokenizer                : /Users/rundel/Scratch/gemma/tokenizer.spm
#> compressed_weights       : /Users/rundel/Scratch/gemma/2b-it-sfp.sbs
#> model                    : "2b-it"
#> weights                  : [no path specified]
#> max_tokens               : 3072
#> max_generated_tokens     : 2048
#> temperature              : 1.000000
#> deterministic            : 0
#> multiturn                : 1
#> verbosity                : 1
#> num_threads              : 8
#> eot_line                 : ""

m$get_config()
#> $tokenizer
#> [1] "/Users/rundel/Scratch/gemma/tokenizer.spm"
#> 
#> $compressed_weights
#> [1] "/Users/rundel/Scratch/gemma/2b-it-sfp.sbs"
#> 
#> $model
#> [1] "2b-it"
#> 
#> $multiturn
#> [1] "1"
#> 
#> $num_threads
#> [1] "8"
```
