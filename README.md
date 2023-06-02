# jointcalib_sigsde

This is a collection of Jupyter notebooks and Python files which have been used in the article:<br><br> 
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


In the present repository you will find the following material. Recall that data were purchased from OptionMetrics and therefore are not present in the current repository. We address the interested reader to our first work for an introduction to signature-based models in mathematical finance (forthcoming in SIAM Journal on Financial Mathematics):

Cuchiero, C.; Gazzani, G.; Svaluto-Ferro, S. Signature-based models: theory and calibration.
```
@article{CGS:22,
  title={{Signature-based models: theory and calibration}},
  author={Cuchiero, C. and Gazzani, G. and Möller, J. and Svaluto-Ferro, S.},
  journal={Preprint arXiv:2207.13136},
  year={2022}
}
```
<div class="about">
                <h2 style="color:#06386D"><b>Sampler for a type of signature-based model</b></h2>
  <ul>
<li>Code for sampling: the Cholesky matrix for the VIX, the linear regression basis on Z for the log-price and the matrix Q0 for the log-price. </li><br>
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
 
  
  
  
  
  
  ![VIX](vix_smiles_calibrated.png)
  <br>
<br>
  <br>
  <br>
    <br>
  <br>
