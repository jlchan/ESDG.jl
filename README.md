# ESDG.jl

A package for creating discretizations of time-dependent hyperbolic PDEs using energy or entropy stable high order discontinuous Galerkin methods on conforming unstructured meshes consisting of triangular, quadrilateral, or hexahedral elements.

These codes are inspired by the Matlab codes for the book [Nodal Discontinuous Galerkin methods](https://link.springer.com/book/10.1007/978-0-387-72067-8) by Hesthaven and Warburton (2007).  

## References
The discretizations used are based on the following references:
- [Nodal discontinuous Galerkin methods](https://link.springer.com/book/10.1007/978-0-387-72067-8)
- [Weight-adjusted discontinuous Galerkin methods: wave propagation in heterogeneous media](https://epubs.siam.org/doi/abs/10.1137/16M1089186?casa_token=j8893ak2KVEAAAAA:wVbmLtWx3ibL03oxn_97uRt7du2cSdf-6XlkHhczsVTmHI_6ndEgHm-fe3W-CmrWKuEf7CEo_i8)
- [On discretely entropy conservative and entropy stable DG methods](https://doi.org/10.1016/j.jcp.2018.02.033)
- [Skew-Symmetric Entropy Stable Modal DG Formulations](https://doi.org/10.1007/s10915-019-01026-w)
- [Efficient Entropy Stable Gauss Collocation Methods](https://doi.org/10.1137/18M1209234)

Special thanks to [Yimin Lin](https://github.com/yiminllin) for providing the initial routines which started this codebase.


