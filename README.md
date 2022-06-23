# DDNet: Drug Drug Interaction Prediction with Meta-Paths and Local Structures

This project is to predict drug drug interactions by an assumption says similar nodes tend to show similar behaviors. Therefore, we set a network and try to represent drugs in new dimensions and also we add topological information by concetenate meta-paths. This github project can be downloaded with 

```
git clone ___
```

### Background
Representational learning is done via node2vec algorithm that uses only nodes' 1-hop or 2 hops neighbors information. The reason why not using entire graph is to decrease complexity and to increase node and graph influence. The way we applied is that:


![Figure_1](https://user-images.githubusercontent.com/37181660/175314027-792efa16-e895-48ad-a182-b4667ea996aa.svg)
<div align="center"> Figure.1 An example of how subgraphs are generated and feature embeddings </div>
<br/>

 - First, we generate subgraphs (1-hop and 2-hops) for each sample
 - Second, subgraphs are given to node2vec algorithm (it is modified word2vec algorithm)
 - Finally, new representaions are obtained in new dimension

Then, we concetenate meta-paths between nodes to add topological information of the drug network to node2vec feature of drugs. These paths are calculated according to length of the paths between two nodes. (Figure 2)

![Figure_2](https://user-images.githubusercontent.com/37181660/175314066-e42a1b6d-5a9f-494d-ba6e-f60e32b54779.svg)
<div align="center"> Figure.2 An example of meta-path features generation and feature vector reconstruction </div>
<br/>


### Use Case

Project guidlines are given in use-cases.

- ***System***: Operating system independent<br>
- ***Required Packages***: numpy-1.18.5, scipy-1.4.1, pandas-1.3.3, scikit-learn-1.0.1, pytorch-1.9.1, rdkit-2021.09.1, node2vec-0.4.5

To install *node2vec* algorithm:
```
pip install node2vec
```


### Data sources
- drug SMILES data: CROssBAR
- drug-drug interaction: DrugBank

>PPI and DTI interaction fetched via CROssBAR Data API<br>
>https://www.ebi.ac.uk/Tools/crossbar/swagger-ui.html

## License

MIT License

DDNet Copyright (C) 2022

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
