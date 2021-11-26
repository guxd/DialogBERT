# DialogBERT

This is a PyTorch implementation of the DialogBERT model described in
[**DialogBERT: Neural Response Generation via Hierarchical BERT with Distributed Utterance Order Ranking**](https://arxiv.org/pdf/2012.01775.pdf).


*** 
 
# Prerequisites
- Python 3.6
- PyTorch

Install packages of the _requirements.txt_ file.


# Usage

- Train model by
  ```
    python main.py
  ```
The logs and temporary results will be printed to stdout and saved in the `./output` path.

- Run test by 
  ```
  python main.py --do_test --reload_from XXXXX
  ```
  where XXXXX specifies the iteration number of the optimal checkpoint.

# License

```
BSD 3-clause

DialogBERT
Copyright 2021-present NAVER Corp.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## References 
If you use any source code included in this toolkit in your work, please cite the following paper:
```
@inproceedings{gu2021dialogbert,
      title={Dialog{BERT}: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances},
      author={Gu, Xiaodong and Yoo, Kang Min and Ha, Jung-Woo},
      journal={In Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI 2021)},
      year={2021}
}
```
