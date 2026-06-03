# An Introduction to Deep Generative Models

**Instructor:** Davide Evangelista (DISI, University of Bologna)  
**Teaching period:** May - June 2026  
**Course year:** First or Second Year PhD  
**Duration:** 12 hours  
**PhD credits:** 2.4  
**Final assessment:** Yes  
**Format:** Hybrid (in person and online via Microsoft Teams)  
**Location:** Bodoniana 2, Via San Donato 19/2, Bologna  

## Course Overview

Deep generative models represent one of the most transformative advancements in artificial intelligence in recent years, due to their remarkable ability to generate images, text, and other forms of data that are often indistinguishable from real-world counterparts. Despite their growing popularity and widespread adoption, these models are frequently treated as black boxes, with limited understanding of the principles that govern their behavior.

This course aims to provide a comprehensive introduction to modern deep generative models, making the subject accessible even to students who do not have prior experience in the field. Throughout the course, both theoretical foundations and practical implementation aspects will be covered, with the goal of enabling students to critically understand and effectively use these models in their own research.

## A First Intuition: The Oracle View of $p_{gt}$

Before introducing architectures, losses, or latent variables, it is useful to build an intuition for the central object of the whole course: the data-generating distribution, denoted by $p_{gt}$. Very informally, $p_{gt}$ is the unknown probabilistic law that produced the images in our dataset. If the dataset contains human faces, then $p_{gt}$ is the distribution of plausible face images. If the dataset contains natural objects, then $p_{gt}$ is the distribution of plausible images of those objects.

A useful divulgative way to think about this is through an **oracle**. Imagine an ideal black box that receives an image $\boldsymbol{x}$ and answers the question: *how compatible is this image with the kind of images that truly come from the world we are modeling?* This oracle does not merely say "real" or "fake". Instead, it assigns high values to images that look like they genuinely belong to the dataset distribution, and low values to images that are visually implausible, corrupted, or simply not representative of the target world.

```{figure} assets/images/Oracle.png
:width: 72%
:align: center

The oracle point of view: a deep generative model tries to approximate the hidden rule that separates highly plausible images from implausible ones.
```

This picture is not a rigorous definition, but it is extremely helpful at the beginning. A **deep generative model** can be viewed as an attempt to approximate this hidden oracle. Of course, we never observe the true function $p_{gt}$ directly. We only observe examples sampled from it. The learning problem is therefore indirect: from many images drawn from the data distribution, we try to construct a model that has captured enough of the same structure to assign high compatibility to plausible images and to generate new images that also look plausible.

## High-Probability and Low-Probability Images

The oracle view makes it easier to understand what people mean when they say that some images have **high probability** under $p_{gt}$ and others have **low probability**.

A high-probability image is not necessarily a repeated or boring image. It is an image that lies in a region of image space where the true data distribution places substantial mass. For a face dataset, this means a face with coherent anatomy, natural texture, realistic lighting, and globally consistent structure. For an object dataset, it means an image whose shape, color, viewpoint, and texture all fit the visual regularities of the dataset.

A low-probability image, by contrast, is not impossible in a mathematical sense. It is simply very unlikely under the true data-generating process. Random static is low probability. So are images with scrambled geometry, unnatural color artifacts, broken object structure, duplicated eyes, smeared boundaries, or severe distortions. These are exactly the kinds of images that often appear when a generative model has not yet learned the data distribution well.

```{figure} assets/images/Sampling.png
:width: 78%
:align: center

An intuitive picture of sampling from $p_{gt}$: high-probability regions correspond to visually coherent images, while low-probability regions correspond to implausible or corrupted ones.
```

This intuition is important because sampling from a generative model can now be described very simply. When we say that a model generates an image, what we really want is that it samples from a distribution close to $p_{gt}$. In the oracle picture, this means that the model should place most of its sampling mass in regions of image space where the oracle would give a strong positive answer. A bad model samples too often from low-probability regions. A good model concentrates its samples in high-probability regions while still preserving diversity.

## Why This View Matters for the Rest of the Course

This oracle-based intuition is the common thread behind all the model families we will study. Variational autoencoders try to learn a structured probabilistic mechanism that reproduces the high-probability regions of the data. GANs train a generator so that its samples become hard to distinguish from true samples. Diffusion models learn how to progressively move noisy images back toward regions of higher probability. Flow matching learns a transport rule that pushes simple randomness into the target distribution.

So even though the methods will look very different mathematically, they are all trying to answer the same foundational question: **how can we learn a model whose samples look as if they were drawn from the true but unknown distribution $p_{gt}$?** Keeping this picture in mind makes the more technical chapters much easier to interpret.

## Learning Objectives

By the end of the course, students will:

- Understand the fundamental concepts underlying deep generative modeling  
- Gain in-depth knowledge of key generative model families, including:
  - Variational Autoencoders (VAEs)
  - Generative Adversarial Networks (GANs)
  - Diffusion Models
  - Flow Matching Models
- Develop the ability to implement these models independently  
- Learn how to integrate generative models into their research workflows  
- Acquire the necessary background to further explore the scientific literature in the field  

## Course Structure

The course is designed to balance theoretical insights with practical components, offering a coherent path from foundational concepts to state-of-the-art generative modeling techniques. Each lecture will combine theoretical explanations with hands-on laboratory sessions, allowing students to directly experiment with the models discussed and to better understand their behavior in practice.

In the first part of the course, we will introduce the probabilistic and mathematical foundations that underlie generative modeling, with particular attention to latent variable models and likelihood-based approaches. These concepts will then be progressively specialized and extended when studying specific model families.

In the subsequent lectures, we will focus on the major classes of deep generative models for image generation, namely Variational Autoencoders, Generative Adversarial Networks, Diffusion Models, and Flow Matching Models. For each of these, we will discuss the theoretical principles, the training procedures, and the main challenges that arise in practice. The laboratory sessions will mirror the theoretical content, guiding students through the implementation of these models and encouraging experimentation.

Throughout the course, emphasis will be placed not only on how these models work, but also on their limitations, assumptions, and appropriate use in research settings. If time allows, we will also explore selected applications of deep generative models, with a particular focus on inverse problems such as super-resolution, image colorization, and deblurring.

## Schedule and Tentative Topics

Each session will include both theoretical and practical components.

- **May 27, 2026 (14:00–17:00)**  
  Introduction to generative modeling, probabilistic foundations, and latent variable models  
  *Lab:* introduction to the coding environment and basic generative modeling examples  

- **June 3, 2026 (14:00–17:00)**  
  Variational Autoencoders (VAEs): theory and training objectives  
  *Lab:* implementation of a VAE for image generation  

- **June 10, 2026 (14:00–17:00)**  
  Generative Adversarial Networks (GANs): architecture, training dynamics, and challenges  
  *Lab:* implementation and experimentation with GANs  

- **June 24, 2026 (14:00–17:00)**  
  Diffusion and Flow Matching models: principles, training procedures, and recent advances  
  *Lab:* implementation of a basic model and discussion of applications  

## Prerequisites

The course is open to PhD students from a wide range of disciplines, including computer science, engineering, mathematics, and related fields. No prior knowledge of deep generative models is required.

A basic familiarity with machine learning concepts, linear algebra, and probability theory is recommended, as these topics will be used throughout the course. Some prior exposure to deep learning, for example through standard neural network architectures and training procedures, may be helpful but is not strictly necessary.

From a practical perspective, students are expected to have basic programming experience in Python. The laboratory sessions will involve implementing models using modern deep learning frameworks, and students should feel comfortable reading and modifying code.

## Assessment

A final assessment is required in order to obtain the PhD credits. The modalities of the exam will be discussed at the beginning of the course, and may include a small project, a practical assignment, or a short report related to the topics covered.

## QrCode for the Course Page

```{figure} assets/images/QRCode.png
:width: 50%
:align: center
```