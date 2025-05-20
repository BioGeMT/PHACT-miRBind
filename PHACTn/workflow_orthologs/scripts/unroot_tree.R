#!/usr/bin/env Rscript

# Set CRAN mirror and load/install ape library
cran_mirror <- "https://cloud.r-project.org"
options(repos = c(CRAN = cran_mirror))
#if (!requireNamespace("ape", quietly = TRUE)) {
  #install.packages("ape", dependencies = TRUE)
#}
library(ape)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Simple argument check
if (length(args) != 2) {
  stop("Usage: Rscript script_name.R <input_tree_file> <output_tree_file>")
}

# Define unrooting function clearly
unroot_tree <- function(tree_file, out_file) {
  tree <- read.tree(tree_file)
  unrooted_tree <- unroot(tree)
  write.tree(unrooted_tree, file = out_file)
}

# Call function with provided arguments
unroot_tree(tree_file = args[1], out_file = args[2])