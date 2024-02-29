
x2 = gemma(
  tokenizer="/Users/rundel/Scratch/gemma/tokenizer.spm",
  compressed_weights="/Users/rundel/Scratch/gemma/2b-it-sfp.sbs",
  model="2b-it",
  verbosity="2"
)

x7 = gemma(
  tokenizer="/Users/rundel/Scratch/gemma/tokenizer.spm",
  compressed_weights="/Users/rundel/Scratch/gemma/7b-it-sfp.sbs",
  model="7b-it",
  verbosity="2"
)

## macbook

z = rgemmacpp:::gemmacpp( list(
  tokenizer="/Users/rundel/Scratch/gemma/tokenizer.spm",
  compressed_weights="/Users/rundel/Scratch/gemma/2b-it-sfp.sbs",
  model="2b-it",
  verbosity="2"
), c("There were only 5 dogs in the park. How many dogs were at the park?",
     "Two of the dogs had to go home, but three new dogs stopped by to play with the others. How many dogs are at the park?")
)


# z = rgemmacpp:::gemmacpp( list(
#   tokenizer="/Users/rundel/Scratch/gemma/tokenizer.spm",
#   compressed_weights="/Users/rundel/Scratch/gemma/7b-it-sfp.sbs",
#   model="7b-it",
#   verbosity="2"
# ),
#  c("There were 5 dogs in the park. How many dogs were at the park?",
#    "In the same park two of the dogs had to go home. How many dogs are there now at the park?")
# )
#
# # remote
#
# z = rgemmacpp:::gemmacpp( list(
#   tokenizer="/data/gemma/build/tokenizer.spm",
#   compressed_weights="/data/gemma/build/7b-it-sfp.sbs",
#   model="7b-it",
#   verbosity="2"
# ),
# c("There were 5 dogs in the park. How many dogs were at the park?",
#   "In the same park two of the dogs had to go home. How many dogs are there now at the park?")
# )
