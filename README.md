# jointcalib_sigsde

This is a collection of Python files which have been used in the article:<br><br> 
"Joint calibration to SPX and VIX options with signature-based models" <br><br>
of <a href ="https://www.mat.univie.ac.at/~cuchiero/">Christa Cuchiero</a>, <a href ="https://homepage.univie.ac.at/guido.gazzani/">Guido Gazzani</a>,  <a href ="https://quarimafi.univie.ac.at/about-us/janka-moeller/">Janka Möller</a> and <a href ="https://sites.google.com/view/sarasvaluto-ferro">Sara Svaluto-Ferro</a>.


For citations:\
**MDPI and ACS Style**\
Cuchiero, C.; Gazzani, G.; Möller J.; Svaluto-Ferro, S. Joint calibration to SPX and VIX options with signature-based models.
```
@article{CGMS:23,
  title={{Joint calibration of SPX and VIX options with signature-based models}},
  author={Cuchiero, C. and Gazzani, G. and Möller, J. and Svaluto-Ferro, S.},
  journal={Preprint arXiv:2301.13235},
  year={2023}
}
```
Click <a href='https://arxiv.org/abs/2301.13235'> here </a> to be redirected to the preprint on ArXiv.
<br>

In the present repository you will find the following material. Recall that data were purchased from OptionMetrics and therefore are not present in the current repository. We address the interested reader to our first work for an introduction to signature-based models in mathematical finance (forthcoming in SIAM Journal on Financial Mathematics):

Cuchiero, C.; Gazzani, G.; Svaluto-Ferro, S. Signature-based models: theory and calibration.
```
@article{CGS:22,
  title={{Signature-based models: theory and calibration}},
  author={Cuchiero, C. and Gazzani, G. and Svaluto-Ferro, S.},
  journal={Preprint arXiv:2207.13136},
  year={2022}
}
```
Click <a href='https://arxiv.org/abs/2207.13136'> here </a> to be redirected to the preprint on ArXiv.

We reference additionally to the Github repository  <a href='https://github.com/sarasvaluto/AffPolySig'> AffPolySig </a>, where a more general implementation of the expected signature of a polynomial process can be found. In particular there one can easily specify the generic parametric form of the desired polynomial diffusion of which the user is interested to compute the time-extended expected signature. Complementary to the approach of polynomial processes, there one can find also the computation of the Laplace transform in the Brownian setting, via the theory of affine processes. For details on the theory we refer to 

Cuchiero, C.; Svaluto-Ferro, S; Teichmann, J.  Signature SDEs from an affine and polynomial perspective.
```
@article{CST:23,
  title={{Signature SDEs from an affine and polynomial perspective}},
  author={Cuchiero, C. and Gazzani, G. and Svaluto-Ferro, S.},
  journal={Preprint arXiv:2302.01362},
  year={2023}
}
```



<div class="about">
                <h2 style="color:#06386D"><b>Sampler for the log-price and the VIX squared</b></h2>
  <ul>
<li>Code for sampling: the Cholesky matrix for the VIX/VIX squared (see Remark 5.5)</li><br>
<li>Code for sampling: the log-price in particular the matrix Q^0 and the regression basis \tilde{e}^{B} (Proposition 6.5, Equation 6.3)</li><br>
        <li>For the VIX squared both numerical integration and exact simulation are reported see Remark 5.4 in the paper. </li><br>
  </ul>
  </div>
  Some comments on the sampler can be found in the paper.
  
  <div class="about">
                <h2 style="color:#06386D"><b>Joint calibration to SPX and VIX options with constant parameters</b></h2>
  <ul>
<li>Code for calibration to option prices of SPX and VIX options.</li><br>
  </ul>
  </div>
  Details of the calibration to option prices with signature-based models can be found in Section 7 of the paper.
 
  
  
  
  ![joint0](joint_calibration_SPX[0,2]_VIX[0]_.png)
  ![joint1](joint_calibration_SPX[4]_VIX[1]_.png)
<br>
  <br>
    <br>
  
  ![VIX](vix_smiles_calibrated.png)
  <br>
<br>
  <br>
  <br>
    <br>
  <br>
