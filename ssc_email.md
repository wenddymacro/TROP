To: kit.baum@bc.edu
Subject: SSC submission: trop

Dear Professor Baum,

I am writing to submit a new package called "trop" for the SSC Archive. The package implements the triply robust panel estimator of Athey, Imbens, Qu & Viviano (2025) for estimating average treatment effects on the treated (ATT) in panel data settings. It combines nuclear-norm regularized outcome modeling, unit/time propensity weighting, and interactive fixed effects to achieve triply-robust inference.

The package has one main estimation command (trop), postestimation diagnostics (estat), and prediction (predict). It includes LOOCV hyperparameter selection, cluster bootstrap inference, covariate adjustment, and event-study analysis. The computational core is implemented in Rust for performance, exposed to Stata via a pre-compiled plugin.

The attached flat ZIP contains all ado-files, help files, a pre-compiled Mata library (ltrop.mlib), and pre-compiled Stata plugins for macOS ARM64, macOS x64, Linux x64, and Windows x64. The package requires Stata 17 or later and has no dependencies on other SSC packages.

Package name: trop
Authors: Xuanyu Cai, Wenli Xu (City University of Macau)
Installation URL: https://raw.githubusercontent.com/gorgeousfish/TROP/main/trop_stata

Thank you for your time.

Best,
Xuanyu Cai
City University of Macau
