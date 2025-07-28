pacman::p_load(lidR)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Arguments have to be <in_root> <out_root> <ground_class> <num_threads>")
}

in_root <- args[1]
out_root <- args[2]
ground_class <- as.integer(strsplit(args[3], ",")[[1]])
num_threads <- as.integer(args[4])

set_lidr_threads(num_threads)

laz_files <- list.files(in_root, pattern = "\\.(laz|las)$", full.names = TRUE, recursive = TRUE)

for (file in laz_files) {
  las <- readLAS(file)
  
  if (is.empty(las)) {
    message("Skipping empty or unreadable file: ", file)
    next
  }
  
  algo <- knnidw()
  tryCatch(
    expr = {
      las <- normalize_height(las, algorithm = algo, use_class = c(ground_class))
    },
    error = function(e){
       cat("Error normalizing ", file, " : ", conditionMessage(e), "\n")
    }
  )
  
  relative_path <- sub(in_root, "", file)
  output_file <- file.path(out_root, relative_path)
  
  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  
  writeLAS(las, output_file)
  rm(las)
  gc()
}

message("Processing complete!")
