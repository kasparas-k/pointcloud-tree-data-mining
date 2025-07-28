pacman::p_load(lidR)

segment_and_save <- function(las_path, out_las_path, algorithm, tree_classes){
    las <- readLAS(las_path)
    if (!missing(tree_classes)){
        las <- filter_poi(las, Classification %in% tree_classes)
    }
    if (algorithm == 'li' | algorithm == 'li2012'){
        algo <- li2012()
    }
    else {
        chm <- rasterize_canopy(las, 0.5, p2r(subcircle = 0.2), pkg = "terra")
        kernel <- matrix(1,3,3)
        chm <- terra::focal(chm, w = kernel, fun = median, na.rm = TRUE)
        ttops <- locate_trees(chm, lmf(5))
        if (algorithm == 'dalponte' | algorithm == 'dalponte2016'){
            algo <- dalponte2016(chm, ttops)
        } else if (algorithm == 'silva' | algorithm == 'silva2016'){
            algo <- silva2016(chm, ttops)
        } else if (algorithm == 'watershed'){
            algo <- watershed(chm)
        } else {
            print(paste("No algorithm choice for ", algorhithm))
            stop()
        }
    }
    las <- segment_trees(las, algo)
    writeLAS(las, out_las_path)
}
