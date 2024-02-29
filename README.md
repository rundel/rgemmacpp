

<!-- README.md is generated from README.Rmd. Please edit that file -->

# rgemmacpp

<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R-CMD-check](https://github.com/rundel/rgemmacpp/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/rundel/rgemmacpp/actions/workflows/R-CMD-check.yaml)
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

r = m$prompt("What are top 5 places I should visit in Durham, NC?")
#> 
#> 1. **Duke University Campus**
#> 2. **Durham Performing Arts Center**
#> 3. **The Duke Homestead**
#> 4. **The Duke Gardens**
#> 5. **The American Tobacco Museum**

r = m$prompt("Which of the previous locations are best for kids?")
#> 
#> The Duke Gardens and the American Tobacco Museum are best for kids. The Duke Gardens offer a variety of activities for kids of all ages, including a children's garden, a playground, and a petting zoo. The American Tobacco Museum offers exhibits on the history of tobacco in Durham, including a factory tour and a museum shop.
```

### 7b model

``` r
m = gemma(
  tokenizer="~/Scratch/gemma/tokenizer.spm",
  compressed_weights="~/Scratch/gemma/7b-it-sfp.sbs",
  model="7b-it",
  multiturn="1"
)

r = m$prompt("What are top 5 places I should visit in Durham, NC?")
#> 
#> 1. **Duke University Campus**
#> 2. **The Museum of Life and Science**
#> 3. **The Duke Gardens**
#> 4. **The North Carolina Museum of Gems**
#> 5. **The Eno River Trail System**

r = m$prompt("Which of the previous locations are best for kids?")
#> 
#> The Museum of Life and Science, the Duke Gardens, and the North Carolina Museum of Gems are all great places for kids.
```
