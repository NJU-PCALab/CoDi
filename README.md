<h1 align="center">
  CoDi:Subject-Consistent and Pose-Diverse Text-to-Image Generation
  <br>
</h1>

<div align="center">

<a href="https://arxiv.org/pdf/2507.08396" style="display: inline-block;">
    <img src="https://img.shields.io/badge/arXiv%20paper-2507.08396-b31b1b.svg" alt="arXiv" style="height: 20px; vertical-align: middle;">
</a>&nbsp;

<a href="https://zhanxin-gao.github.io/CoDi/" style="display: inline-block;">
    <img src="https://img.shields.io/badge/Project_page-Link-green" alt="project page" style="height: 20px; vertical-align: middle;">
</a>&nbsp;

<a href="https://www.youtube.com/watch?v=sSzHHK4q1Dc" style="display: inline-block;">
    <img src="https://img.shields.io/badge/Video-Presentation-blue" alt="project page" style="height: 20px; vertical-align: middle;">
</a>&nbsp;

</div>

<p align="center">
  <a href="#model-architecture">Model Architecture</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> •
  <a href="#acknowledgement">Acknowledgement</a> •
  <a href="#visualization">Visualization</a> 
</p>

[![Teaser](./teaser/teaser.jpeg)](./teaser/teaser.jpeg)

## Model Architecture
[![Architecture](./teaser/architecture.jpg)](./teaser/architecture.jpg)

## How To Use

```bash
# Clone this repository
$ git clone https://github.com/NJU-PCALab/CoDi

# Go into the repository
$ cd CoDi

### Install dependencies ###
$ conda env create --file environment.yml
$ conda activate codi
### Install dependencies ENDs ###

# Run infer code
$ python main.py

# Run benchmark generation code
$ python gen_benchmark.py
```

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{gao2025codi,
  title={Subject-Consistent and Pose-Diverse Text-to-Image Generation},
  author={Gao, Zhanxin and Zhu, Beier and Yao, Liang and Yang, Jian and Tai, Ying},
  journal={arXiv preprint arXiv:2507.08396},
  year={2025}
}
```

## Acknowledgement

We gratefully acknowledge the following repositories for providing useful components and functions that contributed to our work.

- [ConsiStory](https://github.com/NVlabs/consistory)
- [1Prompt1Story](https://github.com/byliutao/1Prompt1Story)

## Visualization

### Qualitative Results
  <figure style="display: inline-block; margin: 20px; text-align: center; max-width: 700px;">
    <img src="./teaser/qualtative_results.jpeg" alt="other_model" style="width: 100%;">
  </figure>

### Long Story Image Generation
<div align="center">
  <figure style="display: inline-block; margin: 20px; text-align: center; max-width: 700px;">
    <img src="./teaser/long_story.jpeg" alt="long_story" style="width: 100%;">
  </figure>



