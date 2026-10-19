#!/usr/bin/env Rscript
# WGCNA module construction (Langfelder & Horvath).
# Input: log-normalized cells × genes. Samples = cells with mobility + RNA + t-type.
# Writes construction tables under the given output directory.

suppressPackageStartupMessages({
  ok <- requireNamespace("WGCNA", quietly = TRUE)
  if (!ok) {
    stop("WGCNA is not installed in this R library. Run revision/scripts/install_wgcna.R")
  }
  library(WGCNA)
})

options(stringsAsFactors = FALSE)
allowWGCNAThreads()

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: q3_wgcna_modules.R <expr_csv> <out_dir> [min_module_size] [seed]")
}
expr_path <- args[[1]]
out_dir <- args[[2]]
min_module_size <- if (length(args) >= 3) as.integer(args[[3]]) else 20L
seed <- if (length(args) >= 4) as.integer(args[[4]]) else 0L
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("Reading expression:", expr_path, "\n")
raw <- read.csv(expr_path, check.names = FALSE)
sample_ids <- as.character(raw[[1]])
datExpr <- as.matrix(raw[, -1, drop = FALSE])
rownames(datExpr) <- sample_ids
storage.mode(datExpr) <- "double"
rm(raw)
cat(sprintf("Input: %d cells × %d genes\n", nrow(datExpr), ncol(datExpr)))

gsg <- goodSamplesGenes(datExpr, verbose = 3)
if (!gsg$allOK) {
  if (sum(!gsg$goodGenes) > 0) {
    cat("Removing genes:", paste(colnames(datExpr)[!gsg$goodGenes], collapse = ", "), "\n")
  }
  if (sum(!gsg$goodSamples) > 0) {
    cat("Removing samples:", paste(rownames(datExpr)[!gsg$goodSamples], collapse = ", "), "\n")
  }
  datExpr <- datExpr[gsg$goodSamples, gsg$goodGenes, drop = FALSE]
}
cat(sprintf("After goodSamplesGenes: %d cells × %d genes\n", nrow(datExpr), ncol(datExpr)))

powers <- 1:20
cat("pickSoftThreshold (unsigned, powers 1:20)…\n")
sft <- pickSoftThreshold(
  datExpr,
  dataIsExpr = TRUE,
  networkType = "unsigned",
  powerVector = powers,
  RsquaredCut = 0.80,
  verbose = 5
)
fit <- sft$fitIndices
write.csv(fit, file.path(out_dir, "wgcna_soft_threshold.csv"), row.names = FALSE)

r2 <- as.numeric(fit$SFT.R.sq)
pwr <- as.integer(fit$Power)
r2[!is.finite(r2)] <- -Inf
reached <- is.finite(r2) & r2 >= 0.80
if (any(reached)) {
  beta <- min(pwr[reached])
  beta_rule <- "lowest power with scale-free R^2 >= 0.80"
} else {
  max_r2 <- max(r2)
  beta <- min(pwr[r2 == max_r2])
  beta_rule <- "no power reached R^2 >= 0.80; lowest power at maximum R^2"
}
r2_used <- as.numeric(fit$SFT.R.sq[fit$Power == beta])
cat(sprintf("Chosen β=%d  scale-free R²=%.4f  (%s)\n", beta, r2_used, beta_rule))
if (!is.na(sft$powerEstimate)) {
  cat(sprintf("WGCNA powerEstimate=%s\n", as.character(sft$powerEstimate)))
}

set.seed(seed)
cat("blockwiseModules (unsigned, minModuleSize=", min_module_size, ")…\n", sep = "")
net <- blockwiseModules(
  datExpr,
  power = beta,
  networkType = "unsigned",
  TOMType = "unsigned",
  minModuleSize = min_module_size,
  reassignThreshold = 0,
  mergeCutHeight = 0.25,
  numericLabels = FALSE,
  pamRespectsDendro = FALSE,
  saveTOMs = FALSE,
  maxBlockSize = max(5000L, ncol(datExpr) + 1L),
  randomSeed = seed,
  verbose = 3
)

module_colors <- net$colors
MEs <- net$MEs
n_grey <- sum(module_colors == "grey")
mod_sizes <- sort(table(module_colors), decreasing = TRUE)
cat("Module sizes:\n")
print(mod_sizes)

# kME vs all eigengenes (including grey if present)
kme <- signedKME(datExpr, MEs, outputColumnName = "kME")

hub_gene <- character(0)
hub_kme <- numeric(0)
mod_names <- character(0)
n_genes <- integer(0)
for (col in names(mod_sizes)) {
  members <- names(module_colors)[module_colors == col]
  kcol <- paste0("kME", col)
  if (!kcol %in% colnames(kme)) {
    # WGCNA names MEgrey -> kMEgrey
    kcol <- paste0("kME", gsub("^ME", "", col))
  }
  if (!kcol %in% colnames(kme)) {
    hits <- grep(paste0(col, "$"), colnames(kme), value = TRUE)
    kcol <- if (length(hits)) hits[[1]] else NA_character_
  }
  vals <- if (!is.na(kcol)) as.numeric(kme[members, kcol]) else rep(NA_real_, length(members))
  i_hub <- if (length(vals) && any(is.finite(vals))) which.max(vals) else 1L
  hub_gene <- c(hub_gene, members[[i_hub]])
  hub_kme <- c(hub_kme, vals[[i_hub]])
  mod_names <- c(mod_names, col)
  n_genes <- c(n_genes, length(members))
}

gene_rows <- data.frame(
  gene = names(module_colors),
  module = unname(module_colors),
  kME = NA_real_,
  hub = 0L,
  stringsAsFactors = FALSE
)
for (i in seq_along(mod_names)) {
  col <- mod_names[[i]]
  kcol <- paste0("kME", col)
  if (!kcol %in% colnames(kme)) {
    hits <- grep(paste0(col, "$"), colnames(kme), value = TRUE)
    kcol <- if (length(hits)) hits[[1]] else NA_character_
  }
  members <- gene_rows$module == col
  if (!is.na(kcol)) {
    gene_rows$kME[members] <- as.numeric(kme[gene_rows$gene[members], kcol])
  }
  gene_rows$hub[gene_rows$gene == hub_gene[[i]]] <- 1L
}
write.csv(gene_rows, file.path(out_dir, "wgcna_module_genes.csv"), row.names = FALSE)

mod_table <- data.frame(
  module = mod_names,
  n_genes = n_genes,
  hub_gene = hub_gene,
  hub_kME = hub_kme,
  is_grey = as.integer(mod_names == "grey"),
  soft_threshold = beta,
  scale_free_R2 = r2_used,
  min_module_size = min_module_size,
  network_type = "unsigned",
  stringsAsFactors = FALSE
)
write.csv(mod_table, file.path(out_dir, "wgcna_module_table.csv"), row.names = FALSE)

me_df <- data.frame(transcriptomics_sample_id = rownames(datExpr), MEs, check.names = FALSE)
write.csv(me_df, file.path(out_dir, "wgcna_module_eigengenes.csv"), row.names = FALSE)

n_modules <- sum(mod_names != "grey")
construction <- data.frame(
  network_type = "unsigned",
  tom_type = "unsigned",
  min_module_size = min_module_size,
  merge_cut_height = 0.25,
  random_seed = seed,
  soft_threshold_beta = beta,
  scale_free_R2 = r2_used,
  beta_selection = beta_rule,
  wgcna_power_estimate = if (is.null(sft$powerEstimate) || is.na(sft$powerEstimate)) NA_real_ else as.numeric(sft$powerEstimate),
  n_cells_input = length(gsg$goodSamples),
  n_genes_input = length(gsg$goodGenes),
  n_cells_used = nrow(datExpr),
  n_genes_used = ncol(datExpr),
  n_genes_removed_gsg = sum(!gsg$goodGenes),
  n_samples_removed_gsg = sum(!gsg$goodSamples),
  n_modules = n_modules,
  n_grey_genes = n_grey,
  wgcna_version = as.character(packageVersion("WGCNA")),
  stringsAsFactors = FALSE
)
write.csv(construction, file.path(out_dir, "wgcna_construction.csv"), row.names = FALSE)

cat(sprintf(
  "Wrote modules: n_modules=%d  n_grey_genes=%d  β=%d  R²=%.4f\n",
  n_modules, n_grey, beta, r2_used
))
cat("Done.\n")
