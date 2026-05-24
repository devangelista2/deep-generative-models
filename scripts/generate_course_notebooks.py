from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


DISPLAY_MATH_RE = re.compile(r"(?ms)^([ \t]*)\$\$\n(.*?)\n\1\$\$$")


def normalize_markdown(text: str) -> str:
    normalized = dedent(text).strip("\n")

    # MyST notebook markdown can misparse raw $$...$$ display blocks into
    # malformed headings or broken theorem content. Converting them to explicit
    # math directives makes the generated HTML much more stable.
    def repl(match: re.Match[str]) -> str:
        indent = match.group(1)
        body = match.group(2)
        body = "\n".join(
            line[len(indent) :] if indent and line.startswith(indent) else line
            for line in body.splitlines()
        )
        return f"{indent}:::{{math}}\n{body}\n{indent}:::"

    return DISPLAY_MATH_RE.sub(repl, normalized)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": normalize_markdown(text).splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    normalized_cells = []
    for idx, cell in enumerate(cells, start=1):
        normalized = dict(cell)
        normalized["id"] = normalized.get("id", f"cell-{idx:04d}")
        normalized_cells.append(normalized)
    return {
        "cells": normalized_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: str, cells: list[dict]) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook(cells), indent=2) + "\n")


def placeholder_cells(title: str, focus: str, next_steps: str) -> list[dict]:
    return [
        md(
            f"""
            # {title}

            This notebook is part of the long-form course build. Its detailed exposition will be expanded in a later pass. The present version records the intended mathematical scope and keeps the Jupyter Book structure stable.

            The focus of this notebook is {focus}. The final version should preserve the same narrative style used in the earlier chapters: long explanatory paragraphs, explicit assumptions, intuitive commentary, and theorem-proof style arguments whenever a core result is stated.
            """
        ),
        md(
            f"""
            ## Planned Development

            {next_steps}
            """
        ),
    ]


write_notebook(
    "00-overview/index.ipynb",
    [
        md(
            r"""
            # An Introduction to Deep Generative Models for Image Generation

            **Instructor:** Davide Evangelista  
            **Institution:** Department of Computer Science and Engineering, University of Bologna  
            **Course type:** PhD course  
            **Duration:** 12 hours  
            **Format:** theory and implementation alternating throughout the course

            Deep generative models are learning systems whose purpose is not only to classify an observation or predict a scalar response, but to model the full distribution of complex objects such as images. This shift in viewpoint is conceptually profound. In discriminative learning we ask for a rule that maps an input to a target. In generative learning we ask for a mechanism that explains how the observed data could have been produced, and once that mechanism has been learned we can sample from it, condition it, compress with it, or use it as a prior in inverse problems. The course is designed for students with heterogeneous backgrounds, so each chapter will develop the material on three intertwined levels: an intuitive narrative, a mathematically precise derivation, and a compact PyTorch implementation that keeps the central ideas visible.

            The main families studied in the course are variational autoencoders, generative adversarial networks, diffusion models, and flow matching. These are not isolated techniques. They are different answers to a common question: how can we describe a complicated distribution on high-dimensional image space using trainable neural networks, and how can we do so in a way that permits learning, sampling, and scientific understanding. The course begins with a short review of neural networks and representation learning, then develops the probabilistic language needed to speak about latent variables and likelihoods, and finally studies the major model families in detail.
            """
        ),
        md(
            r"""
            ## Course Logic

            A useful way to read the course is to keep in mind three recurring objects. The first is the data distribution, denoted by $p_{gt}(\boldsymbol{x})$, which represents the unknown law that generated the images available in the dataset. The second is a trainable model distribution, denoted by $p_\theta(\boldsymbol{x})$, whose parameters are implemented by neural networks. The third is a sampling or inference mechanism, because a generative model is never only a static formula: it must also provide a practical procedure to draw images, infer latent variables, or denoise corrupted observations. Different model classes differ mainly in how they represent these three objects and in which optimization objective makes learning feasible.

            For this reason, the course will repeatedly move between representation, objective, and algorithm. In the variational autoencoder block we shall represent images through latent variables and derive the evidence lower bound as a tractable objective. In the adversarial block we shall replace explicit likelihoods with a game between two neural networks. In the diffusion block we shall transform generation into a sequence of denoising problems and later reinterpret that sequence in continuous time through stochastic differential equations. In the flow matching block we shall learn deterministic transport fields that push a simple distribution toward the data distribution.
            """
        ),
        md(
            r"""
            This progression is intentional rather than historical decoration. The early chapters build the vocabulary needed to see why the later model families differ. Neural-network foundations explain the trainable function classes that all subsequent models use. Probability and latent-variable language explain what is being modeled and why inference becomes difficult. VAEs then show the first disciplined compromise between expressive modeling and tractable learning. GANs expose what changes when one abandons explicit likelihoods. Diffusion and flow matching finally show how a path-based transport viewpoint reorganizes the whole problem. The book is therefore best read as one argument developed in stages, not as a catalog of unrelated architectures.
            """
        ),
        md(
            r"""
            ## Tentative Schedule

            The book is organized as a sequence of chapters rather than as rigid lecture slides, but the intended pacing is the following. The first meetings cover the neural and probabilistic background needed to describe deep generative models without hidden gaps. The next meetings study VAEs and GANs, with particular attention to why their objectives arise and which pathologies they introduce. The last meetings focus on diffusion models and flow matching, where the notation becomes more analytic and the relation between discrete algorithms and continuous probability flows becomes central.

            Because the course is long enough to go beyond a superficial survey, important results will not be quoted and left unexplained. They will be formulated carefully and justified in full detail when the proof is essential for understanding the method. This is particularly important for the derivation of the ELBO, the discriminator optimum in GANs, the variational interpretation of diffusion training, Tweedie's formula, and the relation between stochastic and deterministic samplers.
            """
        ),
        md(
            r"""
            ```{note}
            These notes are intentionally written as long-form lecture material. They are not meant to be terse reference sheets. If a passage feels slower than a slide deck, that is by design: the purpose is to make the mathematical logic readable even to students who have not seen the topic before.
            ```
            """
        ),
    ],
)

write_notebook(
    "intro/environment-setup.ipynb",
    [
        md(
            r"""
            # Environment Setup

            This short notebook collects the practical setup notes for the implementation chapters. The goal is not to prescribe one exact environment, but to make sure that students can run the compact PyTorch notebooks without losing time to avoidable configuration issues during the course.

            At minimum, students should have a working Python installation, Jupyter support, PyTorch, and `torchvision`. A CPU-only environment is sufficient for reading and understanding the notebooks, especially the toy and small-dataset examples. A GPU is helpful but not mandatory for the pedagogical versions included in this book.
            """
        ),
        md(
            r"""
            ## Recommended Packages

            The implementation notebooks assume access to:

            - `python`
            - `jupyter`
            - `torch`
            - `torchvision`
            - `matplotlib`

            The book itself is built with Jupyter Book and the dependencies listed in `requirements.txt`, but students who only want to run the model notebooks do not necessarily need the full documentation toolchain.

            If the goal is to build the book rather than only run the notebooks, install the project requirements inside the active environment with `python -m pip install -r requirements.txt`. In particular, the book configuration enables the `sphinx_proof` extension for theorem and proof blocks, so a partial Jupyter Book installation that omits `sphinx-proof` will fail during `jupyter-book build .`.
            """
        ),
        md(
            r"""
            ## Practical Advice

            Before a live session, it is a good idea to test one small notebook end to end rather than only checking that imports succeed. For example, confirm that tensors can be allocated on the intended device, that `FashionMNIST` downloads correctly, and that plots render in the notebook interface. These small checks are often more informative than a package list alone.

            For the documentation toolchain, a quick sanity check is to run `python -c "import sphinx_proof"` in the same environment where `jupyter-book` lives. If that import fails, the environment does not yet contain the full book-build dependencies even if `jupyter-book` itself is installed.

            It is also worth remembering that the notebooks in this course are teaching notebooks rather than benchmark scripts. If training is slow on local hardware, students can reduce the number of epochs or steps and still learn the main concepts. The important thing is that the code path remains understandable and runnable.
            """
        ),
    ],
)

write_notebook(
    "01-neural-networks/foundations.ipynb",
    [
        md(
            r"""
            # Neural Network Foundations for Image Generation

            The goal of this chapter is not to offer a complete course on deep learning, but to isolate the pieces of neural network theory that are repeatedly used by modern generative models. A generative model for images typically combines function approximation, high-dimensional optimization, structured architectures, and representation learning. Even when the final probabilistic formulation looks sophisticated, the trainable components are still neural networks acting on tensors. If we do not understand why these networks are expressive, how they process spatial information, and why an autoencoder is already a nontrivial unsupervised model, later chapters become unnecessarily opaque.

            We begin from the simplest viewpoint. A neural network is a parameterized function $f_\theta$ obtained by composing affine maps with nonlinearities. The composition principle matters because each layer can be interpreted as a learned change of coordinates. When the input is an image, however, fully connected layers ignore the geometry of the pixel grid. This is why convolutional neural networks, and later U-Nets, became central in generative modeling: they encode locality, translation-aware feature extraction, and a hierarchy of receptive fields. Those architectural biases are not cosmetic. They reduce sample complexity and make the learned model behave in ways that better match images as structured objects.
            """
        ),
        md(
            r"""
            ## Why This Chapter Matters for Generative Modeling

            It is tempting to skip quickly over neural-network preliminaries and move immediately to VAEs, GANs, or diffusion models. For a short seminar that choice might be acceptable. For a real course it is not. Every deep generative model is built from ordinary learnable components whose behavior we must understand well enough to trust their probabilistic interpretation. When a decoder outputs the parameters of a Gaussian distribution, it is still a neural network. When a discriminator distinguishes real from fake, it is still a neural network. When a diffusion model predicts noise, score, or velocity, it is still a neural network. If the student sees only the probabilistic wrapper and not the architectural engine underneath it, the whole subject risks looking more mysterious than it really is.

            There is also a pedagogical reason to slow down here. Students coming from computer science often know that neural networks work in practice but have not reflected much on why certain architectures are repeatedly chosen. Students coming from mathematics may understand the formalism of optimization or probability but may not yet have an intuitive picture of what a convolution actually buys us. Students from less technical backgrounds may have seen images generated by modern AI systems but may still think of the network as an opaque black box. This chapter is therefore meant to give everyone a common vocabulary before the more specialized generative-model chapters begin.
            """
        ),
        md(
            r"""
            ## From Perceptrons to Deep Feature Maps

            Consider an input vector $\boldsymbol{x} \in \mathbb{R}^d$. A single hidden layer network computes
            $$
            f_\theta(\boldsymbol{x}) = \boldsymbol{W}_2 \sigma(\boldsymbol{W}_1 \boldsymbol{x} + \boldsymbol{b}_1) + \boldsymbol{b}_2.
            $$
            This formula already reveals the two ingredients that will remain with us for the whole course. The first is linear mixing through matrices such as $\boldsymbol{W}_1$ and $\boldsymbol{W}_2$. The second is the nonlinear activation $\sigma$, without which the composition would collapse into another affine map. In practice, deep models alternate these operations many times and learn internal representations that become progressively more abstract.

            For image generation, it is useful to remember that neural networks do not discover semantics by magic. They optimize a loss. What changes the nature of the learned representation is the learning problem. A classifier learns features that separate classes. An autoencoder learns features that preserve enough information to reconstruct the input. A denoising network learns features that are predictive of clean structure under corruption. Later chapters will exploit exactly this idea: one can shape a representation by changing the target task, even while keeping similar architectural ingredients.
            """
        ),
        md(
            r"""
            ```{prf:theorem} Universal approximation in a form useful for intuition
            :label: thm-universal-approximation

            Let $K \subset \mathbb{R}^d$ be compact and let $g : K \to \mathbb{R}$ be continuous. For many standard non-polynomial activation functions $\sigma$, a feedforward neural network with one hidden layer can approximate $g$ uniformly on $K$ with arbitrary precision.
            ```

            ```{prf:proof}
            A full proof is beyond the scope of this introductory chapter, but the statement can be justified by a density argument. One shows that finite linear combinations of ridge functions of the form $\sigma(\boldsymbol{w}^\top \boldsymbol{x} + b)$ generate a function class that is dense in $C(K)$ under the sup norm, provided $\sigma$ is not a polynomial and satisfies mild regularity assumptions. The theorem does not claim that the approximation is easy to find, nor that the required width is small. Its value here is conceptual: neural networks are expressive enough in principle, so the real questions concern inductive bias, optimization, and data efficiency.
            ```
            """
        ),
        md(
            r"""
            ## Convolutions, Locality, and Receptive Fields

            Images are arrays, not unordered vectors. A convolutional layer respects this structure by applying the same local filter at many positions. The weight sharing has two consequences. First, the number of parameters is reduced drastically when compared with a dense layer acting on the whole image. Second, the network becomes sensitive to local patterns regardless of where they occur. For classification this means edges, corners, and textures can be recognized across the field of view. For generation it means that local image statistics can be modeled in a coherent way.

            If a stack of convolutional layers is deep enough, the effective receptive field grows, and the model can integrate local evidence into larger structures. This is one reason why modern generative models often combine downsampling and upsampling paths. Fine detail and global organization live at different scales. Architectures that preserve multi-scale information are therefore natural candidates for image synthesis.

            ```{figure} ../assets/images/CNN.png
            :width: 70%
            :align: center

            A convolutional neural network used as a feature extractor. The same local processing idea will later reappear inside encoders, decoders, discriminators, and denoisers.
            ```
            """
        ),
        md(
            r"""
            The phrase *receptive field* deserves more emphasis than it usually receives in quick introductions. If the receptive field is too small, a network may faithfully detect local edges and textures but fail to organize them into a coherent object-level representation. If it is too large too early, the network may mix information too aggressively and lose fine detail. Generative modeling constantly negotiates this tradeoff because image synthesis requires both local realism and global coherence. A face generator must render pores, eyelashes, and hair strands, but it must also place those details consistently relative to eyes, nose, mouth, and head shape. A diffusion denoiser must clean local noise while preserving the large-scale semantic organization of the object. The architecture is therefore part of the modeling hypothesis, not just a computational container.

            There is also a conceptual bridge here to probability. When we say that an architecture has a useful inductive bias, we mean that the class of functions it can represent efficiently aligns with the statistical regularities of the data. Convolutions assume that local patterns are meaningful and recur across positions. Pooling or downsampling assumes that some degree of coarse summary is acceptable at intermediate stages. Skip connections assume that information from earlier layers should sometimes survive almost unchanged into later stages. These are structural assumptions about natural images, expressed not in probabilistic notation but in network design.
            """
        ),
        md(
            r"""
            ```{prf:theorem} Translation equivariance of discrete convolutions
            :label: thm-translation-equivariance

            Let $T_{\boldsymbol{\tau}}$ denote a discrete translation operator acting on an image by shifting it by $\boldsymbol{\tau}$. If $\mathcal{C}$ is a convolution with a fixed kernel and boundary effects are ignored, then
            $$
            \mathcal{C}(T_{\boldsymbol{\tau}}\boldsymbol{x})
            =
            T_{\boldsymbol{\tau}}(\mathcal{C}\boldsymbol{x}).
            $$
            In other words, convolution is translation equivariant.
            ```

            ```{prf:proof}
            A discrete convolution computes each output pixel as the weighted sum of input pixels in a neighborhood determined by the kernel. If the whole input image is shifted by $\boldsymbol{\tau}$, then the neighborhood seen at the translated location is exactly the translated version of the original neighborhood. Because the same kernel weights are reused at every position, the numerical combination performed after translation is identical to the original combination, only relocated by $\boldsymbol{\tau}$. Therefore shifting first and convolving second gives the same result as convolving first and shifting second.
            ```

            The theorem is elementary, but it explains one of the deepest reasons CNNs became standard for images. A learned edge detector should not need to relearn the same edge independently at every location in the image grid. Translation equivariance encodes this prior directly.
            """
        ),
        md(
            r"""
            ## The U-Net as an Information-Preserving Architecture

            The U-Net architecture was originally introduced for image segmentation, yet it became one of the most important backbones in diffusion modeling. Its strength lies in the coexistence of two pathways. The contracting path builds abstract and spatially coarse features. The expanding path reconstructs fine spatial detail. Skip connections transfer intermediate features from encoder to decoder so that localization is not lost.

            This architecture is especially suitable when the output should preserve much of the spatial organization of the input, as happens in denoising and conditional generation. A pure encoder-decoder with a severe bottleneck may discard detail too aggressively. A U-Net mitigates this issue by letting the decoder access multi-scale information directly. From the viewpoint of the later course, the U-Net is less a specific recipe than a structural principle: keep global context and local precision in communication.

            ```{figure} ../assets/images/UNet.png
            :width: 80%
            :align: center

            A U-Net style architecture. In diffusion models, the input image is typically corrupted by noise and the network predicts either the noise, the score, or a related target.
            ```
            """
        ),
        md(
            r"""
            It is useful to compare the U-Net with a classical bottleneck autoencoder. In a strong bottleneck architecture, the network is forced to compress the entire input into a relatively small latent representation before reconstruction begins. This can be very useful when one wants the bottleneck itself to serve as a meaningful latent code, as in a variational autoencoder. In denoising, however, the goal is usually not to compress the whole image into a tiny representation. The goal is to modify the image while preserving its spatial identity. A denoiser benefits from remembering where structures are located. Skip connections provide exactly this memory.

            For non-technical readers, one can imagine the U-Net as a system with two complementary views of the same image. One view zooms out and asks what global structure is present. The other view zooms back in and restores local detail, but it is allowed to consult the earlier high-resolution observations so that fine spatial information is not lost. This is why the architecture appears so naturally in diffusion and restoration tasks. The model repeatedly answers the question: given a partially corrupted image, what should remain and what should be corrected?
            """
        ),
        md(
            r"""
            ## Unsupervised Learning and the Autoencoder Idea

            Before introducing probabilistic latent variable models, it is helpful to study the deterministic autoencoder. An encoder network maps $\boldsymbol{x}$ to a latent code $\boldsymbol{z} = e_\phi(\boldsymbol{x})$, while a decoder reconstructs the image through $\widehat{\boldsymbol{x}} = d_\theta(\boldsymbol{z})$. Training often minimizes a reconstruction loss such as
            $$
            \mathcal{L}_{AE}(\theta, \phi) = \mathbb{E}_{\boldsymbol{x} \sim p_{gt}} \big[\ell(\boldsymbol{x}, d_\theta(e_\phi(\boldsymbol{x})))\big].
            $$
            The attractive idea is that a lower-dimensional latent space may capture the essential factors of variation. Yet, on its own, the deterministic autoencoder does not define a proper generative model unless we also specify how latent codes should be sampled. This observation is one of the conceptual bridges toward the variational autoencoder.

            ```{figure} ../assets/images/AE.png
            :width: 72%
            :align: center

            The deterministic autoencoder suggests the latent-variable viewpoint but does not yet specify a probabilistic prior over latent codes.
            ```
            """
        ),
        md(
            r"""
            The limitation of the deterministic autoencoder should be stated carefully because it is conceptually important for the rest of the course. Suppose we train an excellent encoder-decoder pair with low reconstruction error. We may be tempted to say that we now have a generator: simply sample a latent code and decode it. But sample it from where. The deterministic training objective has not told us which latent codes are likely, which are impossible, or how mass should be distributed in latent space. Some codes may decode to meaningful images, others to nonsense, and nothing in the reconstruction objective alone resolves the ambiguity.

            This is exactly where probabilistic latent-variable models enter the story. They keep the representational appeal of the autoencoder, namely the idea that images may be described through a lower-dimensional latent mechanism, but they add a probabilistic prior and an explicit generative interpretation. The VAE is therefore not an arbitrary variation on the autoencoder theme. It is the precise mathematical answer to the question that the deterministic autoencoder leaves open: how should latent space be organized so that sampling becomes meaningful.
            """
        ),
        md(
            r"""
            The most important takeaway from this chapter is that the architectures used in deep generative models are not arbitrary containers for probability formulas. They encode assumptions about locality, scale, and information flow. When we later introduce encoders, decoders, discriminators, and denoisers, we shall reuse this architectural vocabulary rather than start from scratch each time. Students who want a broader deep learning background may consult the treatments in {cite}`goodfellow2016deep` and {cite}`bishop2006pattern`.
            """
        ),
    ],
)

write_notebook(
    "02-generative-models/introduction.ipynb",
    [
        md(
            r"""
            # Introduction to Generative Models

            A **generative model** is a probabilistic model for data. This sentence is short, but it contains the whole philosophy of the field. If images are viewed as random variables taking values in a high-dimensional space, then a generative model aims to describe the law according to which these images occur. Once such a law has been learned, new images can be sampled, partial information can be completed, missing content can be inferred, and prior knowledge can be injected into downstream tasks. In other words, generation is not only about synthesizing visually pleasing pictures. It is about learning a distribution rich enough to support reasoning.

            Let $\boldsymbol{x} \in \mathbb{R}^d$ denote an image and let $p_{gt}(\boldsymbol{x})$ denote the unknown data-generating distribution. A learned model introduces a family $\{p_\theta(\boldsymbol{x}) : \theta \in \Theta\}$ and tries to choose parameters $\theta$ so that $p_\theta$ is close to $p_{gt}$. The central technical problem is that $p_{gt}$ is never known analytically. We only observe samples $\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(n)}$. Deep generative modeling is therefore the study of how to represent and fit complicated distributions using neural networks from finite data.
            """
        ),
        md(
            r"""
            This chapter sits between the neural-network preliminaries and the probabilistic foundations for a reason. Once we know what kinds of function classes modern deep networks provide, the next question is what those functions are supposed to represent in a generative setting. Before we can talk about latent variables, variational bounds, or diffusion paths, we need a clear picture of the target problem itself. What does it mean to learn a distribution rather than a classifier. Why is image generation harder than label prediction. Which criteria distinguish model families at a high level. Those are the questions this chapter is meant to stabilize.
            """
        ),
        md(
            r"""
            One reason this problem is harder than it may first appear is that high-dimensional image space is dominated by configurations that do not look like natural images at all. If one samples pixel values independently at random, the result is almost always meaningless noise. Natural images form a tiny, highly structured subset of the ambient space, with strong dependencies across locations, scales, and semantic content. A generative model is successful only when it captures these dependencies well enough that fresh samples fall near the same structured region as the data. The challenge is therefore geometric as much as statistical.

            This is also the right place to state an important philosophical point for the course. A generative model is not merely a machine for producing visually plausible outputs. It is a **hypothesis about how data are organized**. When we choose latent variables, adversarial critics, denoising paths, or transport fields, we are choosing not only an optimization objective but also a language for describing image distributions. Different model families correspond to different languages.
            """
        ),
        md(
            r"""
            ## Generative Versus Discriminative Learning

            Discriminative models aim at conditional tasks such as predicting a label $y$ from an image $\boldsymbol{x}$. In probabilistic language, they approximate objects like $p(y | \boldsymbol{x})$ or decision rules derived from it. Generative models target $p(\boldsymbol{x})$ itself, or a joint distribution $p(\boldsymbol{x}, \boldsymbol{z})$ involving hidden variables. This difference changes the geometry of the task. The image space is enormous, multimodal, and structured. Modeling its full law is more ambitious than solving a supervised prediction problem, but it also provides more flexibility.

            A deep generative model is therefore not merely a classical generative model with a large parameter count. The adjective deep signals that the parameterization is compositional and learned by neural networks. The decoder of a VAE, the generator of a GAN, the score network of a diffusion model, and the velocity field of a flow matching model are all examples of neural parameterizations that transform a simple distribution into a complicated one.
            """
        ),
        md(
            r"""
            It is often helpful to contrast the information demanded by the two paradigms. A discriminative classifier may succeed even if it learns only the features necessary to separate categories. It does not need to know how to generate the full image, nor how likely one background texture is relative to another when both correspond to the same label. A generative model, by contrast, must account for the whole observation. It must model nuisance variation, texture, shape, layout, and all the dependencies that make an image look coherent. This is why generative learning is typically harder, but also why it can be more powerful once successful.

            For students who are less familiar with the machine-learning distinction, one simple metaphor can help. A discriminative model acts like an examiner who sees an answer sheet and decides which class it belongs to. A generative model acts like an author who must know how to produce a plausible answer sheet in the first place. The second task contains much richer structural information.
            """
        ),
        md(
            r"""
            ## Why Likelihood Matters, and Why It Is Not the Whole Story

            If a model has a tractable density $p_\theta(\boldsymbol{x})$, a natural learning principle is maximum likelihood: $\theta^\star \in \arg\max_\theta \sum_{i=1}^n \log p_\theta(\boldsymbol{x}^{(i)})$. This objective has a clear statistical interpretation. Under standard assumptions, maximizing empirical log-likelihood amounts to minimizing the Kullback-Leibler divergence from the data distribution to the model family up to an additive constant. The next theorem makes this precise.

            ```{prf:theorem} Maximum likelihood and forward Kullback-Leibler divergence
            :label: thm-mle-kl

            Let $p_{gt}$ be the true data distribution and let $p_\theta$ be any model distribution such that $\log p_\theta(\boldsymbol{x})$ is integrable under $p_{gt}$. Then
            $$
            \operatorname{KL}(p_{gt} \| p_\theta)
            =
            \mathbb{E}_{p_{gt}}[\log p_{gt}(\boldsymbol{x})]
            -
            \mathbb{E}_{p_{gt}}[\log p_\theta(\boldsymbol{x})].
            $$
            Consequently, minimizing $\operatorname{KL}(p_{gt} \| p_\theta)$ over $\theta$ is equivalent to maximizing the expected log-likelihood under the data distribution.
            ```

            ```{prf:proof}
            By definition,
            $$
            \operatorname{KL}(p_{gt} \| p_\theta)
            =
            \mathbb{E}_{p_{gt}}
            \left[
            \log \frac{p_{gt}(\boldsymbol{x})}{p_\theta(\boldsymbol{x})}
            \right].
            $$
            Splitting the logarithm gives
            $$
            \operatorname{KL}(p_{gt} \| p_\theta)
            =
            \mathbb{E}_{p_{gt}}[\log p_{gt}(\boldsymbol{x})]
            -
            \mathbb{E}_{p_{gt}}[\log p_\theta(\boldsymbol{x})].
            $$
            The first term does not depend on $\theta$, so the minimizer of the divergence coincides with the maximizer of the second term. Replacing the population expectation with the empirical average yields the usual maximum likelihood estimator.
            ```
            """
        ),
        md(
            r"""
            The previous argument is elegant, but in deep generative modeling exact likelihood is often unavailable or inconvenient. This is why different model families choose different compromises. Variational autoencoders keep a likelihood but optimize a lower bound because exact inference over latent variables is hard. GANs abandon explicit likelihood and train through adversarial discrimination. Diffusion models recover tractable objectives by decomposing generation into many small denoising steps. Flow matching learns vector fields that transport a simple distribution to the data distribution along a chosen probability path.
            """
        ),
        md(
            r"""
            This variety of strategies is one of the reasons the field can feel fragmented at first. Students often ask whether VAEs, GANs, and diffusion models are competitors, unrelated inventions, or successive improvements of one universal idea. The right answer is more subtle. They are all trying to approximate the unknown data distribution, but they do so by making different tradeoffs between tractable probability, optimization stability, sample quality, and computational cost. A VAE gives us a transparent probabilistic story and amortized inference, but may blur samples. A GAN may produce sharp outputs, but its optimization is delicate and its density is implicit. A diffusion model is often stable and high quality, but sampling may be slow. Flow matching tries to retain the continuous-time transport viewpoint while simplifying training.

            Once these tradeoffs are visible, the course becomes easier to navigate. One is not memorizing isolated acronyms, but studying different answers to the same foundational problem.
            """
        ),
        md(
            r"""
            ## A First Taxonomy

            One can organize the field along two axes. The first axis asks whether the model uses latent variables. The second asks whether the density is evaluated directly, indirectly, or not at all. A VAE introduces a latent variable $\boldsymbol{z}$ and writes $p_\theta(\boldsymbol{x}) = \int p_\theta(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}) \, d\boldsymbol{z}$, but the integral is usually intractable, so learning proceeds by variational inference. A GAN maps a simple latent noise vector to an image through a generator, but the induced density is implicit, meaning that sampling is easy while density evaluation is not. A diffusion model introduces a latent path rather than a single latent code, and its training objective emerges from the controlled corruption and reconstruction of data. Flow matching focuses on continuous transport and learns a deterministic vector field that induces the desired terminal distribution.

            ```{figure} ../assets/images/DLVM.png
            :width: 68%
            :align: center

            The latent-variable viewpoint is one of the recurring conceptual templates of the course.
            ```
            """
        ),
        md(
            r"""
            It is also worth emphasizing that the taxonomy is not rigid. Many modern systems are hybrid. Latent diffusion models combine autoencoding ideas with diffusion in latent space. Conditional GANs and diffusion models add side information and no longer model only a plain marginal distribution. Normalizing flows provide exact likelihoods but are themselves continuous transport models. The purpose of the taxonomy is therefore not to create boxes that every method must fit perfectly. Its purpose is to help the student identify the main design axes: What is the latent structure. Is the model explicit or implicit. Is training likelihood-based, adversarial, denoising-based, or transport-based. How is sampling performed.

            If these questions can be answered for a new paper, then the student has already understood the architecture of the field at a meaningful level.
            """
        ),
        md(
            r"""
            For students who are less comfortable with the formalism, it is helpful to keep an intuitive image in mind. A generative model is an engine that starts from simple randomness and shapes it until it looks like structured data. The technical challenge is to decide how this shaping should be parameterized and how the parameters can be learned from examples. Everything in the rest of the course can be read as a refinement of this simple idea. Classical references that support the broad statistical viewpoint include {cite}`bishop2006pattern`, while the modern deep learning context is discussed in {cite}`goodfellow2016deep`.
            """
        ),
    ],
)

write_notebook(
    "03-probabilistic-foundations/random-variables-and-latent-models.ipynb",
    [
        md(
            r"""
            # Random Variables, Latent Variables, and Deep Latent Variable Models

            Modern generative modeling is impossible to understand well without a comfortable command of basic probability. This chapter therefore pauses before the main model families and develops the probabilistic language that will recur everywhere: joint distributions, marginalization, conditional independence, Bayes' theorem, and latent variable models. The aim is not to turn the course into a full probability textbook, but to make every later derivation readable and conceptually motivated.

            The key modeling move is the introduction of an **unobserved variable** $\boldsymbol{z}$. Instead of trying to model the full complexity of the image distribution directly in pixel space, we imagine that an image $\boldsymbol{x}$ is generated conditionally on some hidden explanatory factors collected in $\boldsymbol{z}$. This does not mean that the world literally contains a clean low-dimensional coordinate system for every dataset, but it captures an idea that is often fruitful in practice: images may lie near a structured set of much lower intrinsic complexity than raw ambient dimension suggests.
            """
        ),
        md(
            r"""
            In terms of course flow, this is the chapter where the philosophical language of generative modeling becomes operational mathematics. The previous chapter argued that a generative model should learn the law of images rather than only a decision boundary. The present chapter now asks how such a law can be factorized, marginalized, conditioned, and inverted. By the end of it, the reader should be ready not just to admire latent-variable models intuitively, but to read their formulas without hidden gaps.
            """
        ),
        md(
            r"""
            There is a pedagogical tension in probability-heavy chapters of machine-learning courses. If the discussion is too short, the later formulas look unmotivated and symbolic. If it is too abstract, students may lose sight of why these ideas matter for images. The right compromise here is to tie every definition to the latent-variable question that motivates the chapter. Why do we care about conditional distributions. Because later we shall write $p_\theta(\boldsymbol{x} | \boldsymbol{z})$. Why do we care about marginals. Because later we shall need to integrate out $\boldsymbol{z}$ to obtain $p_\theta(\boldsymbol{x})$. Why do we care about Bayes' theorem. Because later we shall need to relate the generative direction $p(\boldsymbol{x} | \boldsymbol{z})p(\boldsymbol{z})$ to the inference direction $p(\boldsymbol{z} | \boldsymbol{x})$.

            This chapter should therefore be read less as a detached review of probability and more as the assembly of the language required by the rest of the course. The notation may be classical, but its purpose is modern: to make deep latent-variable models readable rather than magical.
            """
        ),
        md(
            r"""
            ## Joint, Marginal, and Conditional Distributions

            Suppose $\boldsymbol{x}$ and $\boldsymbol{z}$ are random variables with joint density $p(\boldsymbol{x}, \boldsymbol{z})$. From the joint object one obtains the marginal laws by integration:
            $$
            p(\boldsymbol{x}) = \int p(\boldsymbol{x}, \boldsymbol{z}) \, d\boldsymbol{z},
            \qquad
            p(\boldsymbol{z}) = \int p(\boldsymbol{x}, \boldsymbol{z}) \, d\boldsymbol{x}.
            $$
            When $p(\boldsymbol{z}) > 0$, the conditional density is defined by
            $$
            p(\boldsymbol{x} | \boldsymbol{z}) = \frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{z})}.
            $$
            These equations are elementary, but in generative modeling they carry enormous practical meaning. The joint density is often easier to specify than the marginal one. If we choose a prior $p(\boldsymbol{z})$ and a conditional model $p_\theta(\boldsymbol{x} | \boldsymbol{z})$, then the induced data density becomes
            $$
            p_\theta(\boldsymbol{x}) = \int p_\theta(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}) \, d\boldsymbol{z}.
            $$
            This is the canonical latent-variable construction.
            """
        ),
        md(
            r"""
            ```{prf:theorem} Law of total probability for latent-variable models
            :label: thm-total-probability

            Let $\boldsymbol{x}$ and $\boldsymbol{z}$ be random variables with joint density $p(\boldsymbol{x}, \boldsymbol{z})$. Then the marginal density of $\boldsymbol{x}$ satisfies
            $$
            p(\boldsymbol{x}) = \int p(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}) \, d\boldsymbol{z}.
            $$
            ```

            ```{prf:proof}
            Starting from the definition of conditional density,
            $$
            p(\boldsymbol{x}, \boldsymbol{z}) = p(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}).
            $$
            Integrating both sides with respect to $\boldsymbol{z}$ gives
            $$
            \int p(\boldsymbol{x}, \boldsymbol{z}) \, d\boldsymbol{z}
            =
            \int p(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}) \, d\boldsymbol{z}.
            $$
            The left-hand side is precisely the marginal density $p(\boldsymbol{x})$, which proves the result.
            ```

            This theorem is basic probability, but it is also the exact formula that turns latent-variable modeling into a generative model for observations. The whole difficulty of models such as VAEs begins here: the formula is conceptually clean, but the integral is usually intractable once the conditional model is implemented by a flexible neural network.
            """
        ),
        md(
            r"""
            ```{prf:theorem} Bayes' theorem
            :label: thm-bayes

            If $p(\boldsymbol{x}) > 0$, then
            $$
            p(\boldsymbol{z} | \boldsymbol{x}) =
            \frac{p(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z})}{p(\boldsymbol{x})}.
            $$
            ```

            ```{prf:proof}
            By the definition of conditional density,
            $$
            p(\boldsymbol{x}, \boldsymbol{z})
            =
            p(\boldsymbol{z} | \boldsymbol{x}) p(\boldsymbol{x})
            =
            p(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}).
            $$
            Equating the two expressions and dividing by $p(\boldsymbol{x})$ gives the result.
            ```
            """
        ),
        md(
            r"""
            **Bayes' theorem** is the algebraic bridge between generation and inference. The factorization $p(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z})$ describes how data arise from latent causes. The posterior $p(\boldsymbol{z} | \boldsymbol{x})$ describes what can be inferred about those causes once an image has been observed. A large part of modern generative modeling consists of dealing with the fact that the posterior is informative and useful, but often hard to compute exactly.

            A very simple image example can make this less abstract. Imagine that $\boldsymbol{z}$ stores two hidden factors: digit identity and stroke thickness. The decoder answers the forward question: if the hidden factors correspond to "3" with thick strokes, what image is likely to appear. The posterior answers the reverse question: after seeing a specific handwritten image, which hidden identities and stroke styles are plausible explanations. Later models use much richer latent spaces, but the logic is the same.
            """
        ),
        md(
            r"""
            It is worth stressing what Bayes' theorem means operationally. In the generative direction we imagine that hidden causes are sampled first and images appear afterward. In the inferential direction we observe the image and try to reason backward about which hidden causes might have produced it. These two viewpoints are mathematically equivalent because they describe the same joint distribution, but computationally they can be very different. Sampling from the prior and then from the decoder may be easy, while computing the exact posterior may be hard. This asymmetry is one of the defining features of modern deep latent-variable models.

            For intuition, think of a portrait image. The latent variable might encode identity, pose, lighting, hairstyle, or other hidden factors. The conditional model answers the question: if these hidden factors were given, what image would likely be produced. The posterior answers the reverse question: given this observed image, which hidden factors are plausible. Later, the VAE will formalize exactly how to deal with the fact that the reverse question is difficult.
            """
        ),
        md(
            r"""
            ## Conditional Independence and Graphical Structure

            Conditional independence statements are a compact way to encode modeling assumptions. For three random variables $\boldsymbol{x}$, $\boldsymbol{y}$, and $\boldsymbol{z}$, the notation $\boldsymbol{x} \perp \boldsymbol{y} | \boldsymbol{z}$ means that once $\boldsymbol{z}$ is known, additional knowledge of $\boldsymbol{y}$ provides no further information about $\boldsymbol{x}$. In density form, this becomes
            $$
            p(\boldsymbol{x}, \boldsymbol{y} | \boldsymbol{z})
            =
            p(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{y} | \boldsymbol{z}).
            $$
            Probabilistic graphical models visualize these assumptions and guide factorization choices. In the simplest latent-variable graph $\boldsymbol{z} \to \boldsymbol{x}$, the prior on $\boldsymbol{z}$ and the conditional model of $\boldsymbol{x}$ given $\boldsymbol{z}$ fully determine the joint law.

            For intuition, one may think of $\boldsymbol{z}$ as encoding coarse semantic information such as pose, object identity, lighting, or style, and of $\boldsymbol{x}$ as the final image produced from those factors. This interpretation is never exact, but it is helpful. It explains why latent representations are appealing: they may provide a more manageable coordinate system in which the complexity of the data is disentangled or at least organized.
            """
        ),
        md(
            r"""
            A common source of confusion is that graphical models look discrete and symbolic while neural generative models look continuous and high-dimensional. In reality, they are describing complementary aspects of the same model. The graph expresses the dependency structure. The neural network expresses the functional form of a conditional distribution. One should therefore not think of probabilistic graphical models as obsolete toys that were replaced by deep learning. They remain one of the clearest ways to understand what is assumed to depend on what.

            This point matters for model criticism as well. If a generative model fails, the failure may come from many sources: the decoder may be too weak, the optimization may be poor, the latent prior may be badly chosen, or the dependency structure itself may be too naive. Graphical intuition helps us separate these possibilities conceptually.
            """
        ),
        md(
            r"""
            ## The Manifold Hypothesis as a Modeling Heuristic

            The manifold hypothesis informally states that high-dimensional data encountered in practice often concentrate near a set of much lower intrinsic dimension. For images, this means that although a vectorized image may live in $\mathbb{R}^d$ with enormous $d$, only a small subset of configurations correspond to meaningful natural images. Most points in ambient space look like noise. Generative modeling can therefore be interpreted as the attempt to learn the geometry and probability structure of this thin, complicated subset.

            It is important not to overstate the hypothesis. The data need not lie on a perfectly smooth manifold, and realistic datasets are usually noisy, multimodal, and heterogeneous. Still, the heuristic remains useful because it motivates latent variables. If images are controlled by a smaller collection of underlying factors, a latent-variable model may capture them more effectively than a direct unstructured density in pixel space.
            """
        ),
        md(
            r"""
            The manifold hypothesis should be understood as a modeling suggestion, not as dogma. It tells us that high-dimensional data are often structured enough that a lower-dimensional description is meaningful. It does not tell us what the right dimension is, whether the structure is globally smooth, or whether one latent variable is enough. In practice, this is why modern latent-variable models often use moderately large latent spaces, hierarchical latent structures, or richly structured decoders. The hypothesis justifies the search for hidden explanatory coordinates; it does not remove the need for careful model design.
            """
        ),
        md(
            r"""
            ## Deep Latent Variable Models

            A deep latent variable model combines the probabilistic construction of latent variables with the expressive power of neural networks. The generic factorization is
            $$
            p_\theta(\boldsymbol{x}, \boldsymbol{z})
            =
            p_\theta(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}),
            $$
            where $p(\boldsymbol{z})$ is often chosen to be a simple standard Gaussian and the conditional density $p_\theta(\boldsymbol{x} | \boldsymbol{z})$ is parameterized by a decoder network. Learning requires the marginal likelihood
            $$
            p_\theta(\boldsymbol{x}) = \int p_\theta(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}) \, d\boldsymbol{z},
            $$
            while inference requires the posterior $p_\theta(\boldsymbol{z} | \boldsymbol{x})$. These two objects are exactly where the main difficulty lies. The model is expressive because the decoder is a neural network, but that same flexibility generally destroys analytic tractability.

            ```{figure} ../assets/images/DLVM.png
            :width: 70%
            :align: center

            A deep latent variable model combines a simple prior with a flexible neural conditional model.
            ```
            """
        ),
        md(
            r"""
            ```{admonition} Numerical Example: A Handwritten Digit as $(\boldsymbol{x}, \boldsymbol{z})$
            :class: numerical-example

            Let $\boldsymbol{x}$ be the image of a handwritten digit "2". The corresponding latent variable $\boldsymbol{z}$ is not another image, but a hidden description of that drawing. One coordinate of $\boldsymbol{z}$ might encode digit identity, another stroke thickness, another slant, and another whether the bottom loop is open or closed.

            In a generative story, we might first sample a latent description such as $\boldsymbol{z} = (\text{digit}=2,\ \text{thickness}=0.7,\ \text{slant}=-0.2,\ \text{curvature}=0.9)$ and then draw the image from $p_\theta(\boldsymbol{x} | \boldsymbol{z})$. In the inferential story, we observe the actual pixels of a handwritten "2" and ask which hidden descriptions are plausible. The posterior $p_\theta(\boldsymbol{z} | \boldsymbol{x})$ should place high probability on descriptions compatible with a "2" and very low probability on descriptions corresponding to a "7" or to extreme stroke styles not present in the image.

            This example is intentionally informal, but it captures the main idea: $\boldsymbol{x}$ is the visible object, while $\boldsymbol{z}$ is a compressed hidden explanation of why that object looks the way it does.
            ```
            """
        ),
        md(
            r"""
            The variational autoencoder can now be seen as a disciplined answer to this difficulty. Instead of computing the posterior exactly, it introduces an approximating family $q_\phi(\boldsymbol{z} | \boldsymbol{x})$ and optimizes a lower bound to the log-likelihood. This is the first major point where neural networks, probability, and optimization fully meet. Background material for this probabilistic viewpoint may be found in {cite}`bishop2006pattern`, while the deep latent-variable perspective is developed in {cite}`kingma2013auto` and {cite}`rezende2014stochastic`.
            """
        ),
    ],
)

write_notebook(
    "02-generative-models/evaluation-metrics.ipynb",
    [
        md(
            r"""
            # Evaluating Generated Images: FID, KID, and Practical Cautions

            A generative model should not be judged only by a few appealing samples. Human inspection matters, but it is subjective and unstable. A course on deep generative models therefore needs at least a basic language for **quantitative evaluation**. In image generation, two of the most widely discussed metrics are the **Fréchet Inception Distance (FID)** and the **Kernel Inception Distance (KID)**. Both compare statistics of real and generated images after those images have been passed through a pretrained visual feature extractor, typically Inception v3.

            This chapter is placed before the main model families on purpose. Once VAEs, GANs, diffusion models, and flow matching enter the picture, students should already know what it means to say that one model has better coverage, sharper samples, or a lower feature-space discrepancy. Without that vocabulary, later empirical claims remain much harder to interpret.
            """
        ),
        md(
            r"""
            There is also an important methodological lesson here. In supervised learning, evaluation often feels straightforward: compare predictions with labels. In generative modeling there may be no single correct image corresponding to one latent draw, so pixelwise comparison is usually the wrong tool. We need metrics that ask whether the *distribution* of generated images resembles the distribution of real ones. FID and KID are attempts to operationalize exactly that question.
            """
        ),
        md(
            r"""
            ## Why Pixelwise Error Is Usually Not Enough

            Suppose two images depict the same shoe, but one is shifted a few pixels to the left or drawn with slightly different laces. A pixelwise mean squared error can be large even though a human would judge the images as semantically very similar. This makes direct pixel distances a poor proxy for generative quality whenever multiple plausible outputs exist.

            A more useful idea is to compare images in a **feature space** learned by a strong vision network. There, semantically similar images often lie closer together than they do in raw pixel space. FID and KID follow this philosophy. They first map both real and generated images into high-level features, and only then compare the resulting distributions.
            """
        ),
        md(
            r"""
            A simple analogy helps. If one compares two paragraphs letter by letter, a synonym replacement may look like a large error even when the meaning barely changes. If one first maps each paragraph into a semantic embedding, the comparison becomes more meaningful. FID and KID do the image analogue of this move: compare representations rather than raw coordinates.
            """
        ),
        md(
            r"""
            ## Fréchet Inception Distance

            Let $\{\boldsymbol{f}_i^{(r)}\}_{i=1}^{n_r}$ be features extracted from real images and $\{\boldsymbol{f}_j^{(g)}\}_{j=1}^{n_g}$ be features extracted from generated images. FID approximates each feature distribution by a Gaussian:
            $$
            \mathcal{N}(\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r),
            \qquad
            \mathcal{N}(\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g).
            $$
            It then computes the squared Fréchet distance between those two Gaussians:
            $$
            \operatorname{FID}
            =
            \|\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\|_2^2
            +
            \operatorname{tr}
            \Big(
                \boldsymbol{\Sigma}_r
                +
                \boldsymbol{\Sigma}_g
                -
                2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g)^{1/2}
            \Big).
            $$
            Lower values are better.
            """
        ),
        md(
            r"""
            FID is attractive because it reacts to both **mean mismatch** and **covariance mismatch** in feature space. If generated images all look realistic but only cover one narrow subset of the data, the covariance term tends to worsen. If generated images are consistently shifted toward implausible or blurry features, the mean term can worsen. In this sense, FID tries to summarize both quality and diversity in a single scalar.

            The price of this convenience is approximation. The true feature distribution is usually not Gaussian, and FID can be biased for small sample sizes. It is still very useful, but it should never be treated as a perfect oracle.
            """
        ),
        md(
            r"""
            ```{admonition} Numerical Example: Reading FID Intuitively
            :class: numerical-example

            Imagine that real images produce two-dimensional features with mean $\boldsymbol{\mu}_r = (0,0)$ and covariance $\boldsymbol{\Sigma}_r = \boldsymbol{I}$. A generated model produces features with mean $\boldsymbol{\mu}_g = (1,0)$ and the same covariance $\boldsymbol{\Sigma}_g = \boldsymbol{I}$.

            In this toy case, the covariance term cancels out and the FID becomes simply $\|\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\|_2^2 = 1$. So even if the generated distribution has the right spread, a systematic mean shift in feature space still creates a nonzero penalty. If the means matched but the generated covariance collapsed to a much narrower cloud, the covariance term would reveal that lack of diversity instead.
            ```
            """
        ),
        md(
            r"""
            ## Kernel Inception Distance

            KID uses the same general idea of comparing Inception features, but it avoids the Gaussian approximation. Instead, it computes a **maximum mean discrepancy** style comparison between the real and generated feature distributions, usually with a polynomial kernel. In compact notation,
            $$
            \operatorname{KID} = \operatorname{MMD}^2(\mathcal{F}_{real}, \mathcal{F}_{fake}),
            $$
            where the underlying kernel defines how features are compared.

            The practical interpretation is again simple: lower is better. The important conceptual difference is that KID compares distributions through kernel mean embeddings rather than by fitting Gaussians. In the literature, KID is often appreciated because it has an unbiased estimator under finite sampling, whereas FID can be noticeably biased on small datasets.
            """
        ),
        md(
            r"""
            One should not overdramatize the difference. In real projects, researchers often report both metrics because each highlights slightly different aspects of distribution mismatch. FID is historically dominant and easy to compare against prior work. KID can be more trustworthy when sample counts are small. Reporting both is therefore a good habit in a teaching setting because it shows students that evaluation itself has uncertainty and design choices.
            """
        ),
        md(
            r"""
            ## What These Metrics Actually Measure

            FID and KID do **not** directly measure truth, beauty, or usefulness. They measure closeness between feature distributions under a particular pretrained network. This has several consequences.

            First, the metrics are only as meaningful as the feature extractor. If the feature representation is well aligned with the dataset semantics, the scores are often informative. If the dataset is very far from the domain on which the feature extractor was trained, interpretation becomes less reliable.

            Second, a lower FID or KID does not guarantee that every single generated sample looks good. A model may improve its average feature statistics while still occasionally producing artifacts.

            Third, the metrics can be manipulated by implementation details such as sample count, image resizing, preprocessing, and the exact feature layer chosen. For this reason, careful reporting matters.
            """
        ),
        md(
            r"""
            For a PhD audience, the most mature position is neither blind trust nor blanket dismissal. These metrics are useful because they create a common empirical language. They are limited because they compress a complicated perceptual and distributional judgment into a scalar. The right habit is to use them together with visual inspection, task-specific analysis, and transparent reporting of how they were computed.
            """
        ),
        md(
            r"""
            ## Computing FID and KID with `torchmetrics`

            In modern PyTorch workflows, a convenient implementation route is `torchmetrics`. The typical pattern is:

            1. Instantiate the metric object.
            2. Feed batches of real images with `real=True`.
            3. Feed batches of generated images with `real=False`.
            4. Call `.compute()`.

            The most important implementation detail is preprocessing. With the default Inception-based extractor, the metric expects **three-channel RGB images**. Grayscale datasets therefore need channel repetition, and images should be scaled consistently before being passed to the metric.
            """
        ),
        code(
            """
            import torch
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance


            def prepare_for_inception_metrics(images):
                # Expect images in [0, 1] with shape (N, C, H, W).
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                return images.clamp(0.0, 1.0)


            fid = FrechetInceptionDistance(
                feature=2048,
                normalize=True,
                reset_real_features=False,
            ).set_dtype(torch.float64)

            kid = KernelInceptionDistance(
                feature=2048,
                subsets=10,
                subset_size=100,
                normalize=True,
                reset_real_features=False,
            )

            real_batch = prepare_for_inception_metrics(torch.rand(32, 1, 28, 28))
            fake_batch = prepare_for_inception_metrics(torch.rand(32, 1, 28, 28))

            fid.update(real_batch, real=True)
            fid.update(fake_batch, real=False)
            kid.update(real_batch, real=True)
            kid.update(fake_batch, real=False)

            print("FID:", fid.compute().item())
            kid_mean, kid_std = kid.compute()
            print("KID mean:", kid_mean.item(), "KID std:", kid_std.item())
            """
        ),
        md(
            r"""
            The code above is deliberately minimal. In a real evaluation pipeline, one would accumulate many more batches, often cache the real-image features once, and report the sample count together with the final numbers. The implementation notebooks later in the book will reuse exactly this pattern when we evaluate VAE, GAN, and diffusion samples, and they will also explain why the current two-dimensional flow-matching toy example is outside the natural scope of image metrics.
            """
        ),
        md(
            r"""
            ## Final Perspective

            The main purpose of this chapter is not to convince the reader that FID and KID are perfect. It is to establish a disciplined habit: when discussing generated images, separate qualitative judgment from quantitative evidence, and understand what each metric is and is not saying. That habit becomes especially important when comparing model families whose outputs trade off sharpness, coverage, and stability in different ways.
            """
        ),
    ],
)

write_notebook(
    "04-variational-autoencoders/elbo-and-learning.ipynb",
    [
        md(
            r"""
            # Variational Autoencoders: ELBO, Inference, and Learning

            The **variational autoencoder** is one of the clearest examples of how a deep generative model emerges from a probabilistic problem. We start from a latent-variable model $p_\theta(\boldsymbol{x}, \boldsymbol{z}) = p_\theta(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z})$, where $\boldsymbol{z}$ is a latent code and the conditional distribution $p_\theta(\boldsymbol{x} | \boldsymbol{z})$ is implemented by a neural decoder. This model is easy to sample from: first draw $\boldsymbol{z} \sim p(\boldsymbol{z})$, then draw $\boldsymbol{x} \sim p_\theta(\boldsymbol{x} | \boldsymbol{z})$. The challenge is learning, because the marginal likelihood
            $$
            p_\theta(\boldsymbol{x}) = \int p_\theta(\boldsymbol{x} | \boldsymbol{z}) p(\boldsymbol{z}) \, d\boldsymbol{z}
            $$
            is usually intractable, and the exact posterior $p_\theta(\boldsymbol{z} | \boldsymbol{x})$ is rarely available in closed form.

            The VAE addresses this by introducing an encoder, also called a recognition model, $q_\phi(\boldsymbol{z} | \boldsymbol{x})$, whose purpose is to approximate the true posterior. The word variational signals that learning is formulated as an optimization problem over a family of approximate posterior distributions.
            """
        ),
        md(
            r"""
            This is the first chapter where all three strands of the course fully meet: neural parameterization, probabilistic latent-variable modeling, and tractable optimization. The previous chapter ended by identifying the two hard quantities in a deep latent-variable model, namely the marginal likelihood and the exact posterior. The VAE should therefore be read as the first serious answer to that bottleneck. It does not remove the intractability by magic. It reorganizes it into a lower bound that can be optimized with neural encoders and decoders.

            A simple mental picture helps here. Imagine a dataset of shoes. A useful latent code might organize broad factors such as ankle boot versus sandal, heel height, or overall silhouette. The decoder then learns how those hidden factors map back into pixels. The encoder tries to reverse that map from a specific image. The VAE is the mathematical framework that makes this bidirectional story trainable.
            """
        ),
        md(
            r"""
            ## Derivation of the Evidence Lower Bound

            The derivation begins with the identity
            $$
            \log p_\theta(\boldsymbol{x})
            =
            \log \int q_\phi(\boldsymbol{z} | \boldsymbol{x})
            \frac{p_\theta(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z} | \boldsymbol{x})}
            \, d\boldsymbol{z}.
            $$
            Since the logarithm is concave, Jensen's inequality immediately yields a lower bound. This is the central theorem of the chapter.

            ```{prf:theorem} Evidence lower bound
            :label: thm-elbo

            For any density $q_\phi(\boldsymbol{z} | \boldsymbol{x})$ whose support contains the support of $p_\theta(\boldsymbol{z} | \boldsymbol{x})$,
            $$
            \log p_\theta(\boldsymbol{x})
            \geq
            \mathbb{E}_{q_\phi(\boldsymbol{z} | \boldsymbol{x})}
            \left[
            \log p_\theta(\boldsymbol{x}, \boldsymbol{z})
            -
            \log q_\phi(\boldsymbol{z} | \boldsymbol{x})
            \right].
            $$
            The right-hand side is called the evidence lower bound, or ELBO.
            ```

            ```{prf:proof}
            Write
            $$
            \log p_\theta(\boldsymbol{x})
            =
            \log \mathbb{E}_{q_\phi(\boldsymbol{z} | \boldsymbol{x})}
            \left[
            \frac{p_\theta(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z} | \boldsymbol{x})}
            \right].
            $$
            Jensen's inequality for the concave logarithm implies
            $$
            \log p_\theta(\boldsymbol{x})
            \geq
            \mathbb{E}_{q_\phi(\boldsymbol{z} | \boldsymbol{x})}
            \left[
            \log \frac{p_\theta(\boldsymbol{x}, \boldsymbol{z})}{q_\phi(\boldsymbol{z} | \boldsymbol{x})}
            \right].
            $$
            Expanding the logarithm gives the stated formula.
            ```
            """
        ),
        md(
            r"""
            The **ELBO** is more than a technical inequality. It is the training objective that makes the model practical. Expanding the joint density yields
            $$
            \mathcal{L}_{ELBO}(\theta, \phi; \boldsymbol{x})
            =
            \mathbb{E}_{q_\phi(\boldsymbol{z} | \boldsymbol{x})}
            [\log p_\theta(\boldsymbol{x} | \boldsymbol{z})]
            -
            \operatorname{KL}(q_\phi(\boldsymbol{z} | \boldsymbol{x}) \| p(\boldsymbol{z})).
            $$
            The first term is a reconstruction term: it encourages the decoder to explain the observed image well from latent codes sampled from the encoder. The second term regularizes the approximate posterior toward the prior. This is what turns a deterministic autoencoder into a generative model. If the latent codes produced by the encoder remain close to a simple reference distribution, then sampling from that reference distribution at generation time becomes meaningful.
            """
        ),
        md(
            r"""
            ```{prf:theorem} ELBO decomposition with posterior gap
            :label: thm-elbo-gap

            For every observation $\boldsymbol{x}$,
            $$
            \log p_\theta(\boldsymbol{x})
            =
            \mathcal{L}_{ELBO}(\theta, \phi; \boldsymbol{x})
            +
            \operatorname{KL}\big(q_\phi(\boldsymbol{z} | \boldsymbol{x}) \| p_\theta(\boldsymbol{z} | \boldsymbol{x})\big).
            $$
            ```

            ```{prf:proof}
            Starting from the ELBO expression,
            $$
            \mathcal{L}_{ELBO}
            =
            \mathbb{E}_{q_\phi}
            [\log p_\theta(\boldsymbol{x}, \boldsymbol{z}) - \log q_\phi(\boldsymbol{z} | \boldsymbol{x})].
            $$
            Add and subtract $\log p_\theta(\boldsymbol{z} | \boldsymbol{x})$. Using
            $$
            \log p_\theta(\boldsymbol{x}, \boldsymbol{z})
            =
            \log p_\theta(\boldsymbol{z} | \boldsymbol{x}) + \log p_\theta(\boldsymbol{x}),
            $$
            we obtain
            $$
            \mathcal{L}_{ELBO}
            =
            \log p_\theta(\boldsymbol{x})
            -
            \mathbb{E}_{q_\phi}
            \left[
            \log \frac{q_\phi(\boldsymbol{z} | \boldsymbol{x})}{p_\theta(\boldsymbol{z} | \boldsymbol{x})}
            \right].
            $$
            The expectation is exactly the Kullback-Leibler divergence between the approximate and exact posterior, which proves the identity.
            ```
            """
        ),
        md(
            r"""
            This identity explains why the bound is useful. Maximizing the ELBO simultaneously pushes upward the data log-likelihood and pushes downward the gap between the recognition model and the true posterior. The quality of inference and the quality of generation are therefore coupled. If the approximate posterior family is too restrictive, the model may learn a decoder that is easier for the encoder to support rather than one that is genuinely optimal for the data.
            """
        ),
        md(
            r"""
            ```{admonition} Numerical Example: Reading the ELBO Terms
            :class: numerical-example

            Suppose an encoder sees an image of a sneaker and outputs a two-dimensional Gaussian posterior with mean $\boldsymbol{\mu} = (1.2, -0.4)$ and standard deviations $(0.5, 0.8)$. The corresponding variational distribution says that the image is encoded near the point $(1.2, -0.4)$ in latent space, but with noticeable uncertainty, especially in the second coordinate.

            If the decoder reconstructs the sneaker well, the reconstruction term in the ELBO is favorable. But the KL term also checks how far this posterior has drifted from the standard Gaussian prior. Here the first coordinate is shifted away from zero, so there is a regularization cost. The ELBO is therefore balancing two pressures at once: keep enough latent information to reconstruct the sneaker, but do not let the posterior wander arbitrarily far from the simple prior that we want to sample from at generation time.
            ```
            """
        ),
        md(
            r"""
            There is also an important optimization lesson hidden here. The encoder $q_\phi(\boldsymbol{z} | \boldsymbol{x})$ is often called an amortized inference model because a single neural network is trained to solve approximate inference for every observation in the dataset at once. This is computationally powerful, but it also creates an additional approximation layer beyond the choice of variational family itself. The encoder may fail to represent the true posterior not only because the family is too simple, but also because the shared inference network does not allocate enough capacity to every observation equally well.

            For a PhD audience, this is worth stressing because many later improvements to VAEs can be read as interventions on this exact bottleneck. Some methods enrich the variational family, some redesign the decoder likelihood, some modify the objective weighting, and some attack amortization error directly. The classical ELBO is therefore not the end of the story. It is the organizing baseline from which a whole research program begins.
            """
        ),
        md(
            r"""
            ## Gaussian Encoders and the Reparameterization Trick

            In practice, a common choice is
            $$
            q_\phi(\boldsymbol{z} | \boldsymbol{x}) =
            \mathcal{N}(\boldsymbol{z}; \boldsymbol{\mu}_\phi(\boldsymbol{x}), \operatorname{diag}(\boldsymbol{\sigma}_\phi^2(\boldsymbol{x}))).
            $$
            The encoder network outputs the mean vector and the log-variance vector. If the prior is standard Gaussian, the KL term can be computed in closed form. The remaining challenge is differentiating through a stochastic sample from $q_\phi(\boldsymbol{z} | \boldsymbol{x})$. The reparameterization trick solves this by writing
            $$
            \boldsymbol{z}
            =
            \boldsymbol{\mu}_\phi(\boldsymbol{x})
            +
            \boldsymbol{\sigma}_\phi(\boldsymbol{x}) \odot \boldsymbol{\varepsilon},
            \qquad
            \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}).
            $$
            Now the randomness is isolated in $\boldsymbol{\varepsilon}$, which does not depend on $\phi$. The map from parameters to sample becomes differentiable, and gradient-based learning becomes feasible.
            """
        ),
        md(
            r"""
            ```{prf:theorem} Closed-form KL for a diagonal Gaussian encoder
            :label: thm-diagonal-gaussian-kl

            Let
            $$
            q_\phi(\boldsymbol{z} | \boldsymbol{x})
            =
            \mathcal{N}\big(
                \boldsymbol{z};
                \boldsymbol{\mu},
                \operatorname{diag}(\boldsymbol{\sigma}^2)
            \big)
            $$
            and let the prior be
            $$
            p(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}).
            $$
            Then
            $$
            \operatorname{KL}(q_\phi(\boldsymbol{z} | \boldsymbol{x}) \| p(\boldsymbol{z}))
            =
            \frac{1}{2}
            \sum_{j=1}^d
            \left(
                \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1
            \right).
            $$
            ```

            ```{prf:proof}
            By definition,
            $$
            \operatorname{KL}(q \| p)
            =
            \mathbb{E}_q[\log q(\boldsymbol{z}) - \log p(\boldsymbol{z})].
            $$
            For the diagonal Gaussian posterior,
            $$
            \log q(\boldsymbol{z})
            =
            -\frac{1}{2}
            \sum_{j=1}^d
            \left[
                \log(2\pi \sigma_j^2)
                +
                \frac{(z_j-\mu_j)^2}{\sigma_j^2}
            \right],
            $$
            while for the standard Gaussian prior,
            $$
            \log p(\boldsymbol{z})
            =
            -\frac{1}{2}
            \sum_{j=1}^d
            \left[
                \log(2\pi)
                +
                z_j^2
            \right].
            $$
            Subtracting and taking expectation under $q$, we use
            $$
            \mathbb{E}_q[(z_j-\mu_j)^2] = \sigma_j^2,
            \qquad
            \mathbb{E}_q[z_j^2] = \mu_j^2 + \sigma_j^2.
            $$
            The constant $\log(2\pi)$ terms cancel, leaving
            $$
            \operatorname{KL}(q \| p)
            =
            \frac{1}{2}
            \sum_{j=1}^d
            \left(
                \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1
            \right),
            $$
            which proves the formula.
            ```
            """
        ),
        md(
            r"""
            This explicit KL formula is one of the practical reasons the Gaussian VAE became such a standard teaching example. The reconstruction term must still be estimated through stochastic samples, but the regularization term is analytically available. The resulting objective is therefore simple enough to implement directly while still expressing the real variational logic of the model.
            """
        ),
        md(
            r"""
            ## Strengths, Limitations, and the Problem of Blurriness

            VAEs are conceptually elegant, statistically grounded, and usually stable to train. They provide an explicit encoder, which makes latent interpolation and amortized inference natural. Yet they also exhibit limitations. The approximate posterior may be too simple, the decoder likelihood may be misspecified, and pixelwise Gaussian reconstruction losses often encourage averages when multiple plausible outputs exist. This is one reason generated images may appear blurry compared with samples from stronger adversarial or diffusion-based models.

            The correct interpretation of this limitation is subtle. The VAE is not blurry because it is probabilistic. It is blurry because of specific modeling choices, especially the combination of simple variational families and reconstruction likelihoods that reward conditional means. Later improvements therefore act on several fronts: richer posterior families, more expressive decoders, alternative priors, hierarchical latent structures, perceptual losses, and latent-space decoders as in latent diffusion models.

            ```{figure} ../assets/images/VAE_architecture.png
            :width: 76%
            :align: center

            The encoder-decoder structure of the variational autoencoder. The stochastic bottleneck is the feature that turns a deterministic autoencoder into a probabilistic latent-variable model.
            ```
            """
        ),
        md(
            r"""
            Another limitation that deserves explicit mention is **posterior collapse**. If the decoder becomes strong enough to model the data with little dependence on the latent code, the optimizer may drive the approximate posterior toward the prior and effectively ignore $\boldsymbol{z}$. In that regime, the KL term becomes very small, but the latent representation ceases to carry meaningful information. This phenomenon is especially visible in sequence VAEs and in image models with highly expressive decoders, but the conceptual issue is general: the ELBO does not force the model to use the latent variable unless doing so helps the total objective.

            Possible remedies illustrate the broader design space of VAEs. One may weaken or schedule the KL penalty, as in $\beta$-VAE style variants or KL warm-up. One may enrich the posterior family through normalizing flows or hierarchical structure. One may alter the decoder likelihood or use perceptual losses to reduce the incentive toward pixelwise averaging. One may also redesign the architecture so that the decoder cannot ignore the latent code too easily. These are not minor tweaks. They show that once the basic VAE logic is understood, a large family of principled variants becomes accessible.
            """
        ),
        md(
            r"""
            The VAE should be remembered as the canonical bridge between probabilistic latent-variable modeling and neural network training. Its historical formulations were introduced in {cite}`kingma2013auto` and {cite}`rezende2014stochastic`. Many later methods, including diffusion models, can be better understood once one has internalized the VAE logic: choose a latent construction, derive a tractable surrogate objective, and use neural networks to amortize the hard parts.
            """
        ),
    ],
)

write_notebook(
    "04-variational-autoencoders/implementation.ipynb",
    [
        md(
            r"""
            # VAE Implementation Notebook

            The purpose of this notebook is to translate the ELBO derivation into a compact PyTorch implementation that remains readable to students who are seeing the method for the first time. The implementation deliberately stays small. We use grayscale images, a low-dimensional Gaussian latent variable, and a lightweight encoder-decoder pair. The objective is not to compete with industrial image generators, but to make visible the exact correspondence between the mathematics of the previous chapter and the tensors manipulated during training.

            The guiding principle is that every line of code should answer a theoretical question. Where is the approximate posterior $q_\phi(\boldsymbol{z} | \boldsymbol{x})$ represented. Which tensors correspond to the mean and the variance. Where does the reparameterization trick appear. Which term in the code is the reconstruction log-likelihood surrogate, and which one is the Kullback-Leibler regularizer. Once those correspondences are clear, larger implementations become much easier to read.

            This also makes the notebook a useful contrast point for the rest of the course. Compared with later GAN and diffusion implementations, the VAE code is compact enough that students can still hold almost the entire mathematical pipeline in their heads at once. That is why VAEs often remain the best first serious coding laboratory for probabilistic deep generative modeling.
            """
        ),
        md(
            r"""
            ## Imports and Basic Configuration

            The notebook below assumes a standard PyTorch installation together with `torchvision`. The code is intentionally CPU-friendly and uses `FashionMNIST` as a simple image dataset, although the same logic applies to MNIST or to any other small grayscale dataset. The images are normalized to the interval $[0,1]$, which matches the use of a Bernoulli-style reconstruction term implemented through binary cross-entropy on sigmoid outputs.

            The hyperparameters are chosen so the example can produce recognizable samples, not only execute quickly. A latent dimension of size $32$ is still small enough to discuss the bottleneck as an actual geometric constraint, while a convolutional encoder-decoder respects the image grid. These are the kinds of tradeoffs worth naming explicitly in class, because they teach students to see hyperparameters as modeling decisions rather than arbitrary constants copied from a repository.
            """
        ),
        code(
            """
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from pathlib import Path
            from torch.utils.data import DataLoader
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
            from torchvision import datasets, transforms, utils
            from tqdm.auto import tqdm
            import matplotlib.pyplot as plt

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(7)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(7)
            num_workers = 2 if device.type == "cuda" else 0
            project_root = Path.cwd() if (Path.cwd() / "_config.yml").exists() else Path.cwd().parent
            DATA_ROOT = project_root / "data"

            # These settings are still compact, but large enough to produce useful samples.
            batch_size = 128
            latent_dim = 32
            base_channels = 32
            lr = 2e-4
            epochs = 30

            transform = transforms.ToTensor()

            train_dataset = datasets.FashionMNIST(
                root=DATA_ROOT,
                train=True,
                download=True,
                transform=transform,
            )

            test_dataset = datasets.FashionMNIST(
                root=DATA_ROOT,
                train=False,
                download=True,
                transform=transform,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
            """
        ),
        md(
            r"""
            ## Encoder and Decoder

            We use a compact convolutional VAE. This is still simple enough to read in one notebook, but it respects locality and produces much better image samples than a flattened multilayer perceptron trained for a few epochs. The encoder maps the image to two vectors, $\boldsymbol{\mu}_\phi(\boldsymbol{x})$ and $\log \boldsymbol{\sigma}_\phi^2(\boldsymbol{x})$. The decoder maps a latent sample back to image logits, which are finally converted to pixel probabilities through a sigmoid.
            """
        ),
        code(
            """
            class VAE(nn.Module):
                def __init__(self, latent_dim=32, base_channels=32):
                    super().__init__()
                    # 28x28 -> 14x14 -> 7x7.
                    self.encoder = nn.Sequential(
                        nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(base_channels),
                        nn.SiLU(),
                        nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(base_channels * 2),
                        nn.SiLU(),
                        nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
                        nn.BatchNorm2d(base_channels * 4),
                        nn.SiLU(),
                        nn.Flatten(),
                    )
                    encoded_dim = base_channels * 4 * 7 * 7
                    self.mu_head = nn.Linear(encoded_dim, latent_dim)
                    self.logvar_head = nn.Linear(encoded_dim, latent_dim)

                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, encoded_dim),
                        nn.SiLU(),
                        nn.Unflatten(1, (base_channels * 4, 7, 7)),
                        nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(base_channels * 2),
                        nn.SiLU(),
                        nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(base_channels),
                        nn.SiLU(),
                        nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
                    )

                def encode(self, x):
                    h = self.encoder(x)
                    mu = self.mu_head(h)
                    logvar = self.logvar_head(h)
                    return mu, logvar

                def reparameterize(self, mu, logvar):
                    # Sample through parameter-free noise so gradients can flow to mu/logvar.
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + std * eps

                def decode(self, z):
                    logits = self.decoder(z)
                    return logits

                def forward(self, x):
                    mu, logvar = self.encode(x)
                    z = self.reparameterize(mu, logvar)
                    logits = self.decode(z)
                    return logits, mu, logvar


            model = VAE(latent_dim=latent_dim, base_channels=base_channels).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            """
        ),
        md(
            r"""
            The method `reparameterize` is the code-level manifestation of the identity
            $$
            \boldsymbol{z} = \boldsymbol{\mu}_\phi(\boldsymbol{x}) + \boldsymbol{\sigma}_\phi(\boldsymbol{x}) \odot \boldsymbol{\varepsilon},
            \qquad
            \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}).
            $$
            It is worth pausing here pedagogically. Without this rewrite, a direct sample from a parameterized Gaussian would obstruct backpropagation through the stochastic node. After the rewrite, the random source is independent of the learnable parameters, so the computational graph remains differentiable with respect to $\phi$.
            """
        ),
        md(
            r"""
            ## ELBO Loss in Code

            We now implement the negative ELBO. The reconstruction term is approximated by a binary cross-entropy between the reconstructed image probabilities and the input image. This corresponds to choosing a Bernoulli observation model for pixels. The KL term is the closed-form divergence between a diagonal Gaussian posterior and the standard Gaussian prior. Since optimizers minimize rather than maximize, we return the negative ELBO.
            """
        ),
        code(
            """
            def elbo_loss(x, logits, mu, logvar):
                # Bernoulli reconstruction term for pixels in [0, 1].
                reconstruction = F.binary_cross_entropy_with_logits(
                    logits,
                    x,
                    reduction="sum",
                )
                # Closed-form KL for a diagonal Gaussian encoder against N(0, I).
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction + kl
                return loss, reconstruction, kl
            """
        ),
        md(
            r"""
            A subtle but important implementation detail is the use of logits instead of post-sigmoid probabilities inside `binary_cross_entropy_with_logits`. This is numerically more stable than applying a sigmoid manually and then feeding the result to a separate binary cross-entropy function. After training, however, we can still apply a sigmoid for visualization because we want actual pixel intensities in $[0,1]$.

            The KL line in the function is equally important to decode for students. It is not a heuristic penalty chosen because Gaussian latents "feel natural". It is exactly the closed-form expression derived in Theorem {prf:ref}`thm-diagonal-gaussian-kl`, evaluated coordinatewise and summed across the minibatch. One of the nicest moments in teaching VAEs is when students realize that this compact piece of PyTorch code is not an approximation trick layered on top of theory, but the direct computational form of a theorem from the previous notebook.
            """
        ),
        md(
            r"""
            ## Training Loop

            The training loop is intentionally written in explicit form rather than hidden behind utility classes. This is useful in a teaching context because students can see how the VAE objective is accumulated over a minibatch and how the encoder and decoder are updated jointly. The loss is normalized by the dataset size at the end of each epoch so that the reported scale does not depend on the minibatch size.
            """
        ),
        code(
            """
            def train_epoch(model, loader, optimizer, device):
                model.train()
                total_loss = 0.0
                total_reconstruction = 0.0
                total_kl = 0.0

                for x, _ in tqdm(loader, desc="train", leave=False):
                    x = x.to(device)

                    optimizer.zero_grad()
                    logits, mu, logvar = model(x)
                    loss, reconstruction, kl = elbo_loss(x, logits, mu, logvar)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_reconstruction += reconstruction.item()
                    total_kl += kl.item()

                n = len(loader.dataset)
                return {
                    "loss": total_loss / n,
                    "reconstruction": total_reconstruction / n,
                    "kl": total_kl / n,
                }


            @torch.no_grad()
            def evaluate(model, loader, device):
                model.eval()
                total_loss = 0.0
                total_reconstruction = 0.0
                total_kl = 0.0

                for x, _ in tqdm(loader, desc="eval", leave=False):
                    x = x.to(device)
                    logits, mu, logvar = model(x)
                    loss, reconstruction, kl = elbo_loss(x, logits, mu, logvar)

                    total_loss += loss.item()
                    total_reconstruction += reconstruction.item()
                    total_kl += kl.item()

                n = len(loader.dataset)
                return {
                    "loss": total_loss / n,
                    "reconstruction": total_reconstruction / n,
                    "kl": total_kl / n,
                }
            """
        ),
        code(
            """
            history = {"train_loss": [], "train_kl": [], "val_loss": [], "val_kl": []}

            for epoch in tqdm(range(epochs), desc="VAE epochs"):
                train_stats = train_epoch(model, train_loader, optimizer, device)
                val_stats = evaluate(model, test_loader, device)

                history["train_loss"].append(train_stats["loss"])
                history["train_kl"].append(train_stats["kl"])
                history["val_loss"].append(val_stats["loss"])
                history["val_kl"].append(val_stats["kl"])

                print(
                    f"Epoch {epoch + 1:02d} | "
                    f"train loss: {train_stats['loss']:.4f} | "
                    f"train KL: {train_stats['kl']:.4f} | "
                    f"val loss: {val_stats['loss']:.4f} | "
                    f"val KL: {val_stats['kl']:.4f}"
                )
            """
        ),
        md(
            r"""
            Tracking the KL term separately is pedagogically useful. At the beginning of training it may be very small or behave erratically, especially if the decoder learns to ignore the latent code. This phenomenon, often called posterior collapse in stronger sequence and image decoders, is worth discussing explicitly with students because it reveals that optimizing the ELBO involves a delicate balance between reconstruction quality and latent regularization.

            The separate statistics also help diagnose milder pathologies before they become dramatic. If the KL term collapses almost to zero and reconstructions remain sharp, the decoder may be solving too much of the task without using the latent variable. If the KL term becomes very large while reconstructions stay poor, the encoder may be pushing too hard against the prior without learning a useful compression. Reading these quantities together is one of the simplest ways to teach students that the ELBO is not a single opaque number but a negotiated compromise between two modeling objectives.
            """
        ),
        md(
            r"""
            ## Reconstruction, Sampling, and Latent Interpolation

            A generative model should be inspected from several viewpoints. Reconstruction shows whether the encoder-decoder pair preserves enough information. Prior sampling checks whether latent points drawn from $p(\boldsymbol{z})$ decode into plausible images. Latent interpolation reveals whether the learned representation changes smoothly. Each of these probes illustrates a different part of the theory.
            """
        ),
        code(
            """
            @torch.no_grad()
            def show_reconstructions(model, loader, device, n=8):
                model.eval()
                x, _ = next(iter(loader))
                x = x[:n].to(device)
                logits, _, _ = model(x)
                # Sigmoid is only for visualization; training used logits directly.
                recon = torch.sigmoid(logits).view(-1, 1, 28, 28)

                grid = torch.cat([x.cpu(), recon.cpu()], dim=0)
                image = utils.make_grid(grid, nrow=n, pad_value=1.0)
                plt.figure(figsize=(1.5 * n, 3.0))
                plt.imshow(image.permute(1, 2, 0), cmap="gray")
                plt.axis("off")
                plt.show()


            @torch.no_grad()
            def show_samples(model, device, n=16):
                model.eval()
                z = torch.randn(n, latent_dim, device=device)
                logits = model.decode(z)
                samples = torch.sigmoid(logits).view(-1, 1, 28, 28).cpu()
                image = utils.make_grid(samples, nrow=4, pad_value=1.0)
                plt.figure(figsize=(6, 6))
                plt.imshow(image.permute(1, 2, 0), cmap="gray")
                plt.axis("off")
                plt.show()
            """
        ),
        code(
            """
            show_reconstructions(model, test_loader, device)
            show_samples(model, device)
            """
        ),
        code(
            """
            @torch.no_grad()
            def interpolate(model, loader, device, steps=8):
                model.eval()
                x, _ = next(iter(loader))
                x0 = x[0:1].to(device)
                x1 = x[1:2].to(device)

                # Interpolate between posterior means to visualize latent geometry.
                mu0, _ = model.encode(x0)
                mu1, _ = model.encode(x1)

                alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
                z = (1 - alphas) * mu0 + alphas * mu1
                logits = model.decode(z)
                images = torch.sigmoid(logits).view(-1, 1, 28, 28).cpu()

                grid = utils.make_grid(images, nrow=steps, pad_value=1.0)
                plt.figure(figsize=(1.7 * steps, 2.5))
                plt.imshow(grid.permute(1, 2, 0), cmap="gray")
                plt.axis("off")
                plt.show()


            interpolate(model, test_loader, device)
            """
        ),
        md(
            r"""
            Interpolation is often the first moment when students feel the latent-variable idea concretely. If the model has learned a meaningful organization of the data, moving linearly in latent space produces structured and gradual semantic changes in image space. This does not prove that the latent space is disentangled, nor that linear interpolation is the theoretically correct path for every model, but it provides useful qualitative evidence that the prior and the decoder have learned a coherent geometry.
            """
        ),
        md(
            r"""
            ## Quantitative Evaluation with FID and KID

            Visual inspection is essential, but it is not enough on its own. We can also compare VAE samples against real test images using **FID** and **KID**. On a small grayscale dataset such as `FashionMNIST`, the resulting values should be interpreted cautiously: they are useful for relative classroom comparison, not as headline benchmark numbers. Still, they are excellent for teaching how distribution-level evaluation works in practice.

            The default `torchmetrics` implementations use an Inception-style feature extractor, so we must adapt our grayscale images to the expected three-channel format. We also keep the images in the $[0,1]$ range and cache the real features so they do not need to be recomputed if the dataset stays fixed.
            """
        ),
        code(
            """
            def prepare_for_inception_metrics(images):
                # The default Inception feature extractor expects RGB-like inputs.
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                return images.clamp(0.0, 1.0)


            @torch.no_grad()
            def compute_vae_fid_and_kid(model, real_loader, device, num_fake=1000):
                fid = FrechetInceptionDistance(
                    feature=2048,
                    normalize=True,
                    reset_real_features=False,
                ).set_dtype(torch.float64).to(device)
                kid = KernelInceptionDistance(
                    feature=2048,
                    subsets=10,
                    subset_size=100,
                    normalize=True,
                    reset_real_features=False,
                ).to(device)

                # First accumulate features from real images.
                for real_images, _ in tqdm(real_loader, desc="real metrics", leave=False):
                    real_images = prepare_for_inception_metrics(real_images.to(device))
                    fid.update(real_images, real=True)
                    kid.update(real_images, real=True)

                # Then accumulate features from generated samples.
                generated = 0
                pbar = tqdm(total=num_fake, desc="VAE fake metrics", leave=False)
                while generated < num_fake:
                    batch_n = min(batch_size, num_fake - generated)
                    z = torch.randn(batch_n, latent_dim, device=device)
                    logits = model.decode(z)
                    fake_images = torch.sigmoid(logits).view(-1, 1, 28, 28)
                    fake_images = prepare_for_inception_metrics(fake_images)
                    fid.update(fake_images, real=False)
                    kid.update(fake_images, real=False)
                    generated += batch_n
                    pbar.update(batch_n)
                pbar.close()

                kid_mean, kid_std = kid.compute()
                return {
                    "fid": fid.compute().item(),
                    "kid_mean": kid_mean.item(),
                    "kid_std": kid_std.item(),
                }


            metric_scores = compute_vae_fid_and_kid(model, test_loader, device)
            print(metric_scores)
            """
        ),
        md(
            r"""
            This evaluation block also clarifies one of the VAE's characteristic tradeoffs. A VAE may reconstruct cleanly and organize latent space nicely, yet still obtain weaker FID or KID than a sharper generator because the decoded samples can look slightly smooth or blurry in feature space. That does not make the model useless. It means the metric is emphasizing a specific aspect of generative quality, namely closeness of the generated feature distribution to the real one.
            """
        ),
        md(
            r"""
            ## Practical Discussion

            This simple implementation can be extended in several directions. Replacing the MLP with a convolutional encoder-decoder improves image quality and better respects spatial structure. Increasing the latent dimension gives the decoder more expressive freedom, though too large a latent space may weaken regularization. Changing the observation model alters the reconstruction term. For grayscale images in $[0,1]$, Bernoulli decoding is a common pedagogical choice, but Gaussian decoders are also standard in the literature.

            It is also useful to prepare students for what "success" should look like in this notebook. Reconstructions should usually preserve the broad clothing category and silhouette, prior samples should look recognizable though somewhat blurry, and latent interpolations should change shape gradually rather than jump discontinuously. If outputs remain indistinct after several epochs, one should check tensor shapes, loss scaling, and whether the decoder is accidentally applying a sigmoid twice. If reconstructions are excellent but random samples are incoherent, that often signals a weakly organized latent space rather than a complete training failure.

            The deeper lesson is that a VAE is not merely an autoencoder with noise added. It is a probabilistic model whose code reflects a specific decomposition of the log-likelihood surrogate. Once the student can identify the prior, posterior family, reparameterized sample, reconstruction model, and KL regularizer in code, the abstract derivation of the ELBO has already been internalized.
            """
        ),
    ],
)

write_notebook(
    "05-gans/adversarial-learning.ipynb",
    [
        md(
            r"""
            # Generative Adversarial Networks

            Generative adversarial networks offer a radically different answer to the generative modeling problem. Instead of writing a tractable density and optimizing its log-likelihood or a lower bound, GANs learn through competition. A generator network $G_\theta$ transforms a simple latent variable $\boldsymbol{z} \sim p(\boldsymbol{z})$ into a synthetic image $G_\theta(\boldsymbol{z})$. A discriminator network $D_\psi$ receives either a real image or a generated one and outputs a score interpreted as the probability that the input came from the data distribution. Training is then expressed as a two-player game in which the discriminator tries to distinguish real from fake, while the generator tries to fool the discriminator.

            This formulation is historically important because it demonstrated that one can learn a high-quality implicit generative model without evaluating a density explicitly. The induced model distribution is the pushforward of the latent prior through the generator. Sampling is easy. Density evaluation is generally intractable. The training signal comes from the discriminator, which adapts during learning and effectively provides a learned loss.
            """
        ),
        md(
            r"""
            The chapter is placed immediately after VAEs because the contrast is pedagogically sharp. VAEs showed how far one can go by keeping a probabilistic model explicit and paying the price of a variational approximation. GANs ask the opposite question: what happens if we give up tractable likelihoods altogether and instead learn by comparison between real and generated samples. Reading the two chapters back to back is one of the best ways to see that deep generative modeling is not defined by one canonical objective, but by a family of compromises between tractability, sample quality, interpretability, and optimization behavior.
            """
        ),
        md(
            r"""
            There is also a pedagogical reason GANs deserve special care in a PhD course. The basic equations are short enough that students may believe the method is conceptually simpler than VAEs or diffusion models. In one sense that is true: the formalism is compact. In another sense it is misleading, because the compactness hides a difficult optimization problem. GANs teach an important lesson that recurs throughout modern machine learning: a model can be easy to state, elegant in its idealized theory, and still delicate in numerical practice. For this reason, the GAN chapter should be read simultaneously as a study of implicit generative modeling and as a study of what happens when optimization itself becomes part of the model's behavior.

            The contrast with likelihood-based models is especially valuable. In a VAE we had a clearly identified probability model, a lower bound, and an explicit approximate posterior. In a GAN we no longer ask the model to assign a tractable density to an image. Instead, we ask whether generated samples can be distinguished from real ones by a learned critic. The training objective is therefore relational rather than absolute. The generator is judged not by a fixed score function but by the current state of its adversary. This makes the method powerful, but it also means that training dynamics matter as much as the static objective.
            """
        ),
        md(
            r"""
            ## The Minimax Objective

            Let $p_{gt}$ be the data distribution on images and let $p_\theta$ be the distribution induced by $G_\theta(\boldsymbol{z})$ when $\boldsymbol{z} \sim p(\boldsymbol{z})$. The original GAN objective is
            $$
            \min_\theta \max_\psi V(D_\psi, G_\theta)
            =
            \mathbb{E}_{\boldsymbol{x} \sim p_{gt}}[\log D_\psi(\boldsymbol{x})]
            +
            \mathbb{E}_{\boldsymbol{z} \sim p(\boldsymbol{z})}
            [\log(1 - D_\psi(G_\theta(\boldsymbol{z})))].
            $$
            It is useful to interpret the two terms separately. The first rewards the discriminator for assigning large probability to real images. The second rewards it for assigning small probability to generated images. The generator appears only in the second term, where it tries to make synthetic samples difficult to reject.

            From a non-technical perspective, one may think of the discriminator as a continuously improving critic and the generator as a continuously improving counterfeiter. The success of the method comes from the fact that the critic is not fixed. It co-evolves with the generator and therefore provides a task-adaptive loss that is much more informative than a simple pixelwise comparison between a generated image and a single target image.
            """
        ),
        md(
            r"""
            The objective also deserves interpretation as a classification problem. If real and generated samples are mixed with equal prior probability, then the discriminator is solving a binary decision task whose positive class is "came from the data" and whose negative class is "came from the generator." The log terms in the GAN objective are therefore exactly the log-likelihood terms of a probabilistic classifier. This observation is not merely cosmetic. It explains why the discriminator learns quickly at the beginning of training: early fake samples are often easy to reject, so the classification task is simple. The difficulty begins when one asks whether the gradients produced by that classifier remain useful for improving the generator.

            At a deeper level, GANs replace direct probabilistic modeling of $p_{gt}$ by a test of indistinguishability. The generator succeeds when generated samples become hard to separate from real ones by a critic drawn from a rich function class. This connects GANs to a broader family of ideas in statistics and machine learning where distributions are compared through witnesses, critics, or integral probability metrics rather than through tractable density formulas.
            """
        ),
        md(
            r"""
            ```{prf:theorem} Optimal discriminator for a fixed generator
            :label: thm-gan-optimal-discriminator

            Fix a generator and its induced distribution $p_\theta$. For every point $\boldsymbol{x}$ such that $p_{gt}(\boldsymbol{x}) + p_\theta(\boldsymbol{x}) > 0$, the discriminator that maximizes the GAN value function is
            $$
            D^\star(\boldsymbol{x})
            =
            \frac{p_{gt}(\boldsymbol{x})}{p_{gt}(\boldsymbol{x}) + p_\theta(\boldsymbol{x})}.
            $$
            ```

            ```{prf:proof}
            For a fixed generator, the objective can be written pointwise as
            $$
            V(D, G_\theta)
            =
            \int
            p_{gt}(\boldsymbol{x}) \log D(\boldsymbol{x})
            +
            p_\theta(\boldsymbol{x}) \log(1 - D(\boldsymbol{x}))
            \, d\boldsymbol{x}.
            $$
            Since the integrand decouples in $\boldsymbol{x}$, we maximize
            $$
            f(d) = a \log d + b \log(1-d)
            $$
            for $a = p_{gt}(\boldsymbol{x})$ and $b = p_\theta(\boldsymbol{x})$. Differentiating gives
            $$
            f'(d) = \frac{a}{d} - \frac{b}{1-d}.
            $$
            Setting this derivative equal to zero yields
            $$
            a(1-d) = bd,
            \qquad
            d = \frac{a}{a+b}.
            $$
            Because
            $$
            f''(d) = -\frac{a}{d^2} - \frac{b}{(1-d)^2} < 0,
            $$
            this critical point is the unique maximizer. Substituting back $a = p_{gt}(\boldsymbol{x})$ and $b = p_\theta(\boldsymbol{x})$ gives the claimed formula.
            ```
            """
        ),
        md(
            r"""
            The formula for $D^\star$ is revealing. An optimal discriminator does not memorize examples individually. It compares the relative mass that the real and generated distributions assign to each region of image space. Regions where the generator oversamples relative to the data are penalized, and regions where the data dominate are rewarded. This is the first hint that adversarial learning is implicitly minimizing a divergence between distributions rather than matching samples one by one.

            It is worth pausing here because this theorem often changes how students think about GANs. The discriminator is sometimes described informally as "detecting realism," but the formula is more precise. It says that the ideal discriminator estimates a density ratio. In regions where the generator places too much mass relative to the data, the discriminator moves downward. In regions where the data dominate, it moves upward. The generator therefore receives a signal about where its distribution is misallocated, even though no explicit density is ever evaluated. This is one of the most conceptually beautiful aspects of the GAN framework.
            """
        ),
        md(
            r"""
            ```{prf:theorem} The GAN game and Jensen-Shannon divergence
            :label: thm-gan-jsd

            Let $D^\star$ be the optimal discriminator for a fixed generator. Then
            $$
            V(D^\star, G_\theta)
            =
            -\log 4 + 2 \operatorname{JSD}(p_{gt} \| p_\theta),
            $$
            where $\operatorname{JSD}$ denotes the Jensen-Shannon divergence. Consequently, if the discriminator is optimal at each step, minimizing the GAN objective with respect to the generator amounts to minimizing the Jensen-Shannon divergence between the data and model distributions.
            ```

            ```{prf:proof}
            Substitute the optimal discriminator into the value function:
            $$
            V(D^\star, G_\theta)
            =
            \int p_{gt}(\boldsymbol{x})
            \log \frac{p_{gt}(\boldsymbol{x})}{p_{gt}(\boldsymbol{x}) + p_\theta(\boldsymbol{x})}
            \, d\boldsymbol{x}
            +
            \int p_\theta(\boldsymbol{x})
            \log \frac{p_\theta(\boldsymbol{x})}{p_{gt}(\boldsymbol{x}) + p_\theta(\boldsymbol{x})}
            \, d\boldsymbol{x}.
            $$
            Let
            $$
            m(\boldsymbol{x}) = \frac{1}{2}(p_{gt}(\boldsymbol{x}) + p_\theta(\boldsymbol{x})).
            $$
            Then
            $$
            \log \frac{p_{gt}(\boldsymbol{x})}{p_{gt}(\boldsymbol{x}) + p_\theta(\boldsymbol{x})}
            =
            \log \frac{p_{gt}(\boldsymbol{x})}{2m(\boldsymbol{x})}
            =
            \log \frac{p_{gt}(\boldsymbol{x})}{m(\boldsymbol{x})} - \log 2,
            $$
            and similarly for the second term. Therefore
            $$
            V(D^\star, G_\theta)
            =
            \operatorname{KL}(p_{gt} \| m)
            +
            \operatorname{KL}(p_\theta \| m)
            -
            2 \log 2.
            $$
            By definition,
            $$
            \operatorname{JSD}(p_{gt} \| p_\theta)
            =
            \frac{1}{2}\operatorname{KL}(p_{gt} \| m)
            +
            \frac{1}{2}\operatorname{KL}(p_\theta \| m).
            $$
            Hence
            $$
            V(D^\star, G_\theta)
            =
            2 \operatorname{JSD}(p_{gt} \| p_\theta) - \log 4.
            $$
            This proves the claim.
            ```
            """
        ),
        md(
            r"""
            The theorem is elegant, but it must be interpreted carefully. In practice the discriminator is never optimized exactly, the generator and discriminator are finite neural networks, and stochastic gradient descent only approximates the game dynamics. The Jensen-Shannon interpretation is therefore a guiding idealization rather than a literal description of every training step. Still, it explains why GANs are not arbitrary heuristics. They instantiate a principled distribution-matching game.

            This is also the right place to explain why the theory can look stronger than the practice. The Jensen-Shannon divergence is perfectly meaningful when both distributions are fixed objects and the discriminator is optimized over all measurable functions. But actual GAN training lives in a restricted and moving landscape: the discriminator is a finite neural network, the generator changes after every update, minibatches add stochastic noise, and the two players may learn on different time scales. As a consequence, one should think of the theorem as describing the ideal geometry that motivates the method, not as proving that every practical run of SGD is faithfully minimizing Jensen-Shannon divergence step by step.
            """
        ),
        md(
            r"""
            A further subtlety becomes important in high-dimensional image generation. When the supports of $p_{gt}$ and $p_\theta$ are far apart or lie on thin lower-dimensional sets, the Jensen-Shannon divergence can become locally uninformative. Intuitively, if the discriminator can separate real and fake samples almost perfectly, then its output may saturate near zero or one on most of the relevant regions, and the generator may receive poor directional guidance. This is one of the reasons the original GAN formulation produced stunning visual results but also inspired a large stabilization literature. The issue is not that the theory is wrong. The issue is that the geometry induced by Jensen-Shannon divergence may be awkward for gradient-based optimization when distributions overlap weakly.
            """
        ),
        md(
            r"""
            ## Why the Original Generator Loss Can Saturate

            If we follow the minimax formulation literally, the generator minimizes
            $$
            \mathbb{E}_{\boldsymbol{z} \sim p(\boldsymbol{z})}
            [\log(1 - D_\psi(G_\theta(\boldsymbol{z})))].
            $$
            Early in training, however, the discriminator often becomes very good before the generator has learned anything meaningful. In that regime, $D_\psi(G_\theta(\boldsymbol{z}))$ may be close to zero, and the gradient received by the generator can become weak or poorly conditioned. For this reason, practical GAN training often replaces the minimax generator loss with the non-saturating alternative
            $$
            \mathcal{L}_{G,\mathrm{NS}}(\theta)
            =
            - \mathbb{E}_{\boldsymbol{z} \sim p(\boldsymbol{z})}
            [\log D_\psi(G_\theta(\boldsymbol{z}))].
            $$
            This alternative has the same fixed point, in the sense that it still encourages generated samples to move toward regions that the discriminator classifies as real, but it tends to provide stronger gradients when the generator is weak.
            """
        ),
        md(
            r"""
            The word *saturate* deserves a more explicit explanation than it often receives in brief presentations. Suppose the generator is poor and the discriminator is already very confident, so that $D_\psi(G_\theta(\boldsymbol{z})) \approx 0$ for most latent samples. In the minimax loss, the quantity $\log(1 - D_\psi(G_\theta(\boldsymbol{z})))$ is then close to $\log 1 = 0$, and the derivative that reaches the generator may become very small after passing through the discriminator's nonlinearities. The problem is not simply that the loss value is large or small. The problem is that the gradient field can become weak exactly when the generator most needs useful corrective information.

            The non-saturating loss changes the emphasis. Instead of minimizing $\log(1 - D_\psi(G_\theta(\boldsymbol{z})))$, it maximizes $\log D_\psi(G_\theta(\boldsymbol{z}))$ or equivalently minimizes its negative. When the discriminator is confident that fake samples are fake, this objective penalizes the generator more aggressively and often produces stronger gradients. The fixed point is unchanged in the idealized setting because both objectives encourage fake samples to move toward regions judged real. But the local training dynamics can be much better. This is a central example of a broader principle in deep learning: mathematically equivalent goals at equilibrium can behave very differently under gradient descent.
            """
        ),
        md(
            r"""
            One can push this lesson further. GAN research repeatedly shows that choosing a useful optimization surrogate matters as much as choosing the final objective being approximated. Two players may share the same desired equilibrium while differing substantially in whether their gradients guide them there efficiently. This perspective helps students connect GANs to later topics such as score matching and flow matching, where the practical success of the method also depends on rewriting a difficult objective into a regression problem with better gradients.
            """
        ),
        md(
            r"""
            ## Mode Collapse

            One of the central pathologies of GAN training is mode collapse. The data distribution may have many distinct modes corresponding, for instance, to different object classes, poses, or textures. A generator trained adversarially can sometimes discover that producing only a small subset of visually convincing outputs is sufficient to fool the current discriminator. As a result, sample quality may appear high while sample diversity is poor.

            This phenomenon reflects the fact that the adversarial game is a dynamic optimization problem rather than a static convex program. The discriminator reacts to the generator, the generator reacts to the discriminator, and the resulting vector field in parameter space can exhibit oscillations, local instabilities, and partial equilibria that do not represent the full data distribution well. In lectures, this is an excellent place to emphasize that a powerful objective on paper does not automatically imply easy optimization in practice.
            """
        ),
        md(
            r"""
            A helpful intuition is to imagine a dataset containing several well-separated semantic groups, such as shoes, bags, and coats. If the current discriminator is not yet sensitive to the absence of some groups, the generator may learn that producing only one or two convincing categories yields a strong short-term reward. Once that happens, the discriminator eventually adapts, but the generator may then jump toward a different narrow subset rather than spreading its mass more evenly across the full data distribution. The resulting training trajectory can oscillate between partial solutions instead of converging to broad coverage.

            This interpretation is useful because it prevents a common misunderstanding. Mode collapse is not simply "the generator repeats the same picture" in a trivial memorization sense. It is a distributional failure in which the generator allocates probability mass too narrowly relative to the diversity of the data. The samples may still look sharp and varied at first glance, especially to a casual observer. The failure becomes visible only when one asks whether the whole dataset is represented fairly. This is one reason evaluation in GANs is subtle: visual inspection alone is informative but not sufficient.
            """
        ),
        md(
            r"""
            Several remedies can be interpreted as attempts to give the discriminator a smoother or more globally meaningful signal. Feature matching encourages the generator to match intermediate discriminator statistics rather than only the final decision boundary. Minibatch discrimination lets the discriminator look at relationships among samples in a batch so it can detect suspicious lack of diversity. Wasserstein objectives change the underlying geometry so that moving mass between modes produces a more graded penalty. None of these ideas completely solves adversarial optimization, but together they illustrate the central point that diversity failures are tied to the structure of the game, not merely to insufficient model capacity.
            """
        ),
        md(
            r"""
            ## Variants and Stabilization Strategies

            A large part of the GAN literature can be read as an attempt to stabilize the adversarial game or to replace the underlying divergence with a better behaved discrepancy. Wasserstein GANs replace the Jensen-Shannon geometry with the Wasserstein-1 distance, leading to smoother gradients when the supports of the data and generator distributions barely overlap. Spectral normalization constrains the Lipschitz constant of the discriminator layer by layer and often improves stability dramatically with very little implementation overhead. CycleGAN extends the adversarial principle to unpaired image-to-image translation by combining adversarial losses with a cycle-consistency penalty that encourages the forward and backward mappings to preserve content.

            ```{figure} ../assets/images/GAN_architecture.png
            :width: 76%
            :align: center

            A standard GAN architecture with a generator and a discriminator trained in opposition.
            ```
            """
        ),
        md(
            r"""
            Wasserstein GAN is especially important because it changes not only a training heuristic but the geometry of the problem. The Wasserstein-1 distance measures the cost of transporting probability mass from the model distribution to the data distribution. Unlike Jensen-Shannon divergence, it can vary continuously even when the distributions have nearly disjoint support. This is exactly the regime one often faces early in GAN training. The Kantorovich-Rubinstein duality then rewrites the transport distance as a supremum over 1-Lipschitz critics, which explains why Lipschitz control becomes central in WGAN training.

            The original WGAN paper enforced the Lipschitz constraint by weight clipping, which made the idea historically influential but also introduced optimization pathologies of its own. Later work used gradient penalties or spectral normalization to impose smoother constraints. From a teaching perspective, this is a beautiful example of an abstract mathematical condition becoming a concrete implementation problem. One first proves that a certain function class gives the right dual formulation, and then one must still decide how a neural network should be constrained to live approximately inside that class.
            """
        ),
        md(
            r"""
            Spectral normalization deserves separate emphasis because it is one of the simplest practically successful stabilizers. If each linear map in the discriminator is divided by an estimate of its largest singular value, then the operator norm of that layer is controlled. While the global Lipschitz constant of a deep network is more subtle than a product of per-layer constants, this procedure often gives a strong enough approximation to improve stability substantially. Students sometimes appreciate spectral normalization because it shows that not every important GAN idea requires rewriting the whole objective; sometimes a carefully chosen architectural constraint changes the training dynamics dramatically.

            CycleGAN illustrates a different axis of variation. The adversarial mechanism is not limited to unconditional generation from noise. It can be embedded in structured tasks such as image-to-image translation. When paired data are unavailable, adversarial losses alone do not determine a unique semantic mapping between domains. The cycle-consistency term addresses this by requiring that a sample translated from domain A to domain B and back again should recover the original content approximately. The resulting model is a reminder that GANs are best understood as a training principle that can be combined with additional inductive biases and reconstruction terms, not only as one standalone architecture.
            """
        ),
        md(
            r"""
            The conceptual position of GANs within the course is now clear. VAEs emphasized explicit probabilistic structure and a tractable lower bound. GANs sacrifice explicit likelihood evaluation and instead learn through an adaptive critic. Diffusion models will later revisit explicit probabilistic reasoning, but through denoising trajectories rather than adversarial games. The original GAN formulation is due to {cite}`goodfellow2014generative`, while influential stabilization directions include {cite}`arjovsky2017wasserstein`, {cite}`miyato2018spectral`, and {cite}`zhu2017unpaired`.
            """
        ),
    ],
)

write_notebook(
    "05-gans/implementation-and-variants.ipynb",
    [
        md(
            r"""
            # GAN Implementation and Variants

            This notebook translates the adversarial formulation into a compact PyTorch implementation. As in the VAE notebook, the purpose is pedagogical clarity rather than state-of-the-art performance. We use a simple multilayer perceptron generator and discriminator on `FashionMNIST`, which is enough to expose the alternating optimization pattern, the role of latent noise, and the difference between discriminator and generator objectives. Once these mechanics are clear, convolutional GANs and stronger image synthesis pipelines become much easier to read.

            The main conceptual difference from the VAE implementation is that there is no explicit likelihood or ELBO to evaluate. Instead, the training loop alternates between two objectives. The discriminator updates its parameters to separate real and generated images. The generator then updates its parameters so that the current discriminator assigns large realism scores to generated samples. This means that the loss is not static: it changes throughout training because the critic itself is learned.
            """
        ),
        md(
            r"""
            For live teaching, it is useful to state the expectations up front. This notebook is not meant to produce state-of-the-art samples, and it is not a benchmark recipe. Its purpose is to make the adversarial mechanics visible enough that students can connect the code directly to the theory chapter. In particular, the alternating optimization pattern, the use of detached fake samples during the discriminator update, and the distinction between minimax and non-saturating generator training are all easier to understand in code than in equations alone.

            The notebook also provides a good setting for discussing why GAN engineering became such a large subfield. Even with a small MLP and a small dataset, tiny changes in normalization, optimizer settings, network balance, or data scaling can alter the training behavior noticeably. Rather than hiding this fragility, we will use it as part of the lesson. Students should leave the notebook understanding not only how to implement a GAN, but also why adversarial optimization is more temperamental than standard supervised learning.
            """
        ),
        md(
            r"""
            ## Imports and Dataset

            We again choose `FashionMNIST` because it is small, grayscale, and diverse enough to show both partial success and partial failure. This is useful in class because students can see that adversarial training sometimes produces sharp local texture earlier than a VAE, while still suffering from instability and imperfect coverage of the data distribution.
            """
        ),
        code(
            """
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from pathlib import Path
            from torch.utils.data import DataLoader
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
            from torchvision import datasets, transforms, utils
            from tqdm.auto import tqdm
            import matplotlib.pyplot as plt

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(7)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(7)
            num_workers = 2 if device.type == "cuda" else 0
            project_root = Path.cwd() if (Path.cwd() / "_config.yml").exists() else Path.cwd().parent
            DATA_ROOT = project_root / "data"

            # DCGAN-style settings are still teachable but produce much better samples.
            latent_dim = 128
            base_channels = 64
            batch_size = 128
            lr = 2e-4
            epochs = 50

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ])

            train_dataset = datasets.FashionMNIST(
                root=DATA_ROOT,
                train=True,
                download=True,
                transform=transform,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
            """
        ),
        md(
            r"""
            The hyperparameters here are intentionally conservative for a convolutional GAN. A latent dimension of 128 is large enough to permit variety without making the generator unnecessarily hard to train in a classroom example. The learning rate and Adam coefficients follow long-standing GAN heuristics rather than a theorem. This is worth saying explicitly: part of GAN literacy is learning to distinguish which choices come from mathematical derivation and which come from community practice.
            """
        ),
        md(
            r"""
            ## Generator and Discriminator

            The generator maps latent noise $\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ to an image-shaped tensor. The discriminator receives an image and returns a scalar logit. We do not apply a final sigmoid inside the discriminator because `binary_cross_entropy_with_logits` is numerically more stable when working directly with logits. In the generator we apply a final `tanh`, so the dataset transform rescales training images from $[0,1]$ to $[-1,1]$.
            """
        ),
        code(
            """
            class Generator(nn.Module):
                def __init__(self, latent_dim=128, base_channels=64):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(latent_dim, base_channels * 4 * 7 * 7),
                        nn.BatchNorm1d(base_channels * 4 * 7 * 7),
                        nn.ReLU(True),
                        nn.Unflatten(1, (base_channels * 4, 7, 7)),
                        nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(base_channels * 2),
                        nn.ReLU(True),
                        nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(base_channels),
                        nn.ReLU(True),
                        nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
                        nn.Tanh(),
                    )

                def forward(self, z):
                    return self.net(z)


            class Discriminator(nn.Module):
                def __init__(self, base_channels=64):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(base_channels * 2),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Flatten(),
                        nn.Linear(base_channels * 2 * 7 * 7, 1),
                    )

                def forward(self, x):
                    return self.net(x)


            G = Generator(latent_dim=latent_dim, base_channels=base_channels).to(device)
            D = Discriminator(base_channels=base_channels).to(device)

            g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
            d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
            """
        ),
        md(
            r"""
            The choice of Adam with $\beta_1 = 0.5$ is a traditional GAN heuristic. It is not a theorem, but a practical convention that often helps stabilize the adversarial updates. The deeper pedagogical point is that GANs are sensitive to optimization details to a much greater extent than plain supervised models. This is one reason why code examples matter so much in this part of the course.

            The architecture is still simple enough that each block remains interpretable, but it uses the spatial inductive bias that image generation needs. This is a better classroom default than a flattened MLP because students can see the adversarial game while the samples are also good enough to inspect seriously.
            """
        ),
        md(
            r"""
            ## Adversarial Losses

            We use the non-saturating generator loss discussed in the theory notebook. The discriminator is trained with binary cross-entropy to classify real images as real and generated images as fake. The generator is trained to maximize the probability that generated images are classified as real. In code, this means that generated images are paired with a target label of one during the generator step.
            """
        ),
        code(
            """
            def discriminator_loss(real_logits, fake_logits):
                # Mild one-sided label smoothing usually stabilizes the discriminator.
                real_targets = torch.full_like(real_logits, 0.9)
                fake_targets = torch.zeros_like(fake_logits)

                real_loss = F.binary_cross_entropy_with_logits(real_logits, real_targets)
                fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
                return real_loss + fake_loss


            def generator_loss(fake_logits):
                # Non-saturating loss: the generator wants fake samples judged as real.
                real_targets = torch.ones_like(fake_logits)
                return F.binary_cross_entropy_with_logits(fake_logits, real_targets)
            """
        ),
        md(
            r"""
            This pair of functions maps directly onto the theory. The discriminator loss combines one term for real images and one for fake images. The generator loss is the non-saturating alternative, so fake logits are compared against a target of one. In words, the generator is trained as if its own samples ought to be classified as real. Students often find this clearer when said explicitly: the generator never sees real images directly in its loss. It only sees how the discriminator currently reacts to its own outputs.

            If one wanted to implement the literal minimax generator objective instead, the code would be very short to change. The theory notebook explains why we usually do not make that change in practice. This is an excellent point to encourage students to modify the loss and compare the training behavior themselves later, even if the main course notes avoid turning every notebook into an exercise sheet.
            """
        ),
        md(
            r"""
            ## Training Loop

            The alternating structure is the essential new ingredient. For each minibatch, we first update the discriminator using both real and fake samples. The fake samples are detached during the discriminator step so that gradients do not flow back into the generator prematurely. We then sample a fresh batch of latent vectors and update the generator using the current discriminator. This is a simple instance of a two-time-scale game, even though we use one discriminator step and one generator step per batch here.
            """
        ),
        code(
            """
            history = {"d_loss": [], "g_loss": []}


            fixed_z = torch.randn(16, latent_dim, device=device)

            for epoch in tqdm(range(epochs), desc="GAN epochs"):
                d_running = 0.0
                g_running = 0.0

                for real_images, _ in tqdm(train_loader, desc="train", leave=False):
                    real_images = real_images.to(device)
                    batch_n = real_images.size(0)

                    # First update the discriminator on real and detached fake samples.
                    z = torch.randn(batch_n, latent_dim, device=device)
                    fake_images = G(z)

                    d_optimizer.zero_grad()
                    real_logits = D(real_images)
                    fake_logits = D(fake_images.detach())
                    d_loss = discriminator_loss(real_logits, fake_logits)
                    d_loss.backward()
                    d_optimizer.step()

                    # Then update the generator against the current discriminator.
                    z = torch.randn(batch_n, latent_dim, device=device)
                    fake_images = G(z)

                    g_optimizer.zero_grad()
                    fake_logits = D(fake_images)
                    g_loss = generator_loss(fake_logits)
                    g_loss.backward()
                    g_optimizer.step()

                    d_running += d_loss.item()
                    g_running += g_loss.item()

                d_epoch = d_running / len(train_loader)
                g_epoch = g_running / len(train_loader)
                history["d_loss"].append(d_epoch)
                history["g_loss"].append(g_epoch)

                print(
                    f"Epoch {epoch + 1:02d} | "
                    f"D loss: {d_epoch:.4f} | "
                    f"G loss: {g_epoch:.4f}"
                )
            """
        ),
        md(
            r"""
            There are several details here that are worth highlighting aloud in a lecture. First, we draw a fresh latent batch for the generator step instead of reusing the exact same noise vectors as in the discriminator step. This is not logically necessary, but it makes the two updates conceptually distinct. Second, the call to `detach()` is crucial. Without it, the discriminator update would also backpropagate into the generator graph, mixing the roles of the two players. Third, the discriminator sees image tensors directly because the convolutional architecture uses spatial structure.

            More advanced GAN implementations often use unequal numbers of discriminator and generator updates, gradient penalties, exponential moving averages of generator weights, or larger batches. These additions can matter a great deal in realistic image synthesis. The minimal loop here should therefore be read as the bare skeleton of adversarial optimization, not as a universal recipe.
            """
        ),
        md(
            r"""
            These losses must be interpreted with care. In supervised learning, lower validation loss usually has a fairly stable meaning. In GANs, discriminator and generator losses can oscillate, flatten, or even move in apparently counterintuitive directions while sample quality changes. This is because the objective is relative: a strong discriminator can make the generator loss high even when the samples are improving, and vice versa. For that reason, one should always inspect generated images rather than rely only on scalar losses.

            A useful classroom warning is that a discriminator loss near zero is not automatically a victory, and a generator loss that increases is not automatically a disaster. If the discriminator becomes too strong too early, the generator may stop receiving helpful information. Conversely, a temporary rise in generator loss may simply mean that the critic has improved its standards. Scalar curves still matter, but mainly as contextual signals rather than as standalone performance scores. The most trustworthy diagnosis usually combines loss trajectories, visual samples, and knowledge of the training setup.
            """
        ),
        md(
            r"""
            ## Sampling and Visual Inspection

            The simplest diagnostic is to sample from the generator at regular intervals and inspect the resulting image grid. Since the generator outputs values in $[-1,1]$, we rescale them back to $[0,1]$ for display. In a classroom setting it is useful to compare these images against the VAE samples from the previous notebook and discuss the tradeoff between sharpness, stability, and coverage.
            """
        ),
        code(
            """
            @torch.no_grad()
            def show_gan_samples(generator, device, n=16):
                generator.eval()
                z = fixed_z[:n] if n <= fixed_z.size(0) else torch.randn(n, latent_dim, device=device)
                samples = generator(z).view(-1, 1, 28, 28)
                # Undo tanh scaling for display.
                samples = 0.5 * (samples + 1.0)
                image = utils.make_grid(samples.cpu(), nrow=4, pad_value=1.0)
                plt.figure(figsize=(6, 6))
                plt.imshow(image.permute(1, 2, 0), cmap="gray")
                plt.axis("off")
                plt.show()
                generator.train()


            show_gan_samples(G, device)
            """
        ),
        code(
            """
            plt.figure(figsize=(7, 4))
            plt.plot(history["d_loss"], label="discriminator")
            plt.plot(history["g_loss"], label="generator")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            When this notebook works reasonably well, one should expect the generator to move from amorphous noise toward recognizable clothing silhouettes, often with sharper edges than a comparably small VAE produces. At the same time, one should also expect imperfections: repeated shapes, missing classes, unstable textures, or sensitivity to random initialization. Those imperfections are not incidental. They are part of what the notebook is meant to teach.

            If training fails badly, a few debugging clues are especially common. If all samples look nearly identical, suspect mode collapse or an overpowered discriminator. If both losses oscillate wildly and samples never improve, learning rates may be too large or the network balance may be poor. If samples remain pure noise while the discriminator loss drops immediately, the discriminator may be learning too fast relative to the generator. These are not exhaustive diagnoses, but they help students see adversarial training as something one interprets dynamically rather than passively runs.
            """
        ),
        md(
            r"""
            ## Quantitative Evaluation with FID and KID

            GANs are the model family where quantitative image metrics often become especially important. A GAN can produce very sharp samples while still missing modes of the true data distribution. FID and KID help us ask whether the generated image *distribution* resembles the real one, not merely whether a few hand-picked samples look good.

            As in the VAE notebook, the values below should be treated as classroom diagnostics rather than benchmark-grade results. The dataset is small, the architecture is intentionally minimal, and the generated images are grayscale, so we adapt them to the expected three-channel format before feature extraction.
            """
        ),
        code(
            """
            def prepare_for_inception_metrics(images):
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                return images.clamp(0.0, 1.0)


            @torch.no_grad()
            def compute_gan_fid_and_kid(generator, real_loader, device, num_fake=1000):
                fid = FrechetInceptionDistance(
                    feature=2048,
                    normalize=True,
                    reset_real_features=False,
                ).set_dtype(torch.float64).to(device)
                kid = KernelInceptionDistance(
                    feature=2048,
                    subsets=10,
                    subset_size=100,
                    normalize=True,
                    reset_real_features=False,
                ).to(device)

                for real_images, _ in tqdm(real_loader, desc="real metrics", leave=False):
                    real_images = prepare_for_inception_metrics(0.5 * (real_images.to(device) + 1.0))
                    fid.update(real_images, real=True)
                    kid.update(real_images, real=True)

                generated = 0
                pbar = tqdm(total=num_fake, desc="GAN fake metrics", leave=False)
                while generated < num_fake:
                    batch_n = min(batch_size, num_fake - generated)
                    z = torch.randn(batch_n, latent_dim, device=device)
                    fake_images = generator(z).view(-1, 1, 28, 28)
                    # Undo tanh scaling so the metric sees images in [0, 1].
                    fake_images = 0.5 * (fake_images + 1.0)
                    fake_images = prepare_for_inception_metrics(fake_images)
                    fid.update(fake_images, real=False)
                    kid.update(fake_images, real=False)
                    generated += batch_n
                    pbar.update(batch_n)
                pbar.close()

                kid_mean, kid_std = kid.compute()
                return {
                    "fid": fid.compute().item(),
                    "kid_mean": kid_mean.item(),
                    "kid_std": kid_std.item(),
                }


            metric_scores = compute_gan_fid_and_kid(G, train_loader, device)
            print(metric_scores)
            """
        ),
        md(
            r"""
            This metric block is especially useful for discussing **mode collapse**. A GAN may improve visually while still obtaining disappointing FID or KID if it allocates mass too narrowly over the dataset. That makes these scores a nice complement to image grids: the image grid shows local sharpness, while FID and KID say something more about global coverage.
            """
        ),
        md(
            r"""
            ## Variant Notes

            The implementation above is intentionally minimal, but it also exposes the main weaknesses of the vanilla GAN. If training oscillates or samples collapse to a narrow subset of patterns, this is not an accident but a characteristic of the original adversarial game. Three influential corrections are worth keeping in mind.

            In Wasserstein GANs, the discriminator is replaced by a critic that outputs unrestricted real values rather than probabilities. The training objective changes so that the critic approximates the Wasserstein-1 distance between real and generated distributions. This often produces smoother gradients, especially when the two supports overlap poorly. In spectral normalization GANs, each linear or convolutional layer of the discriminator is normalized by an estimate of its spectral norm. This controls the layer-wise Lipschitz constant and tends to improve stability with minimal code changes. In CycleGAN, the adversarial mechanism is used in a conditional translation setting where one learns mappings between two domains without paired examples; the cycle-consistency term is then what prevents arbitrary mappings that fool the discriminator but destroy semantic content.
            """
        ),
        md(
            r"""
            From the perspective of code evolution, these variants tell a useful story. WGAN changes the loss and the meaning of the critic output. Spectral normalization changes the architecture while leaving much of the outer training loop intact. CycleGAN changes the task itself by introducing domain-conditioned generators, two discriminators, and additional reconstruction structure. Seeing these as different axes of modification helps students avoid the impression that every new GAN paper is a totally unrelated invention. Many are targeted interventions on one of three objects: the objective, the critic class, or the problem structure.
            """
        ),
        md(
            r"""
            The deeper lesson from this notebook is that GANs are simple to state but delicate to optimize. The code is shorter than a VAE implementation because there is no explicit inference model or ELBO, yet the practical behavior is often harder to control. This is an excellent opportunity to emphasize to students that conceptual brevity and optimization ease are not the same thing.
            """
        ),
    ],
)

write_notebook(
    "06-diffusion-models/discrete-diffusion.ipynb",
    [
        md(
            r"""
            # Discrete Diffusion Models

            **Diffusion models** return us to an explicitly probabilistic style of reasoning, but they do so in a very different way from VAEs. Instead of introducing a single latent code $\boldsymbol{z}$ that explains an image, we introduce an entire latent trajectory $\boldsymbol{x}_0, \boldsymbol{x}_1, \ldots, \boldsymbol{x}_T$, where $\boldsymbol{x}_0$ is the clean data point and the later variables are progressively corrupted versions of it. The key idea is that while direct generation of $\boldsymbol{x}_0$ is difficult, generation may become much easier if we learn to invert a sequence of small corruption steps. Each reverse transition only needs to remove a little bit of noise, and this local denoising problem turns out to be remarkably well suited to neural networks.

            In this chapter we focus on the discrete-time viewpoint that leads to denoising diffusion probabilistic models, or DDPMs. Later chapters will reinterpret the same ideas in continuous time using stochastic differential equations. For now, the discrete presentation is valuable because it exposes the relation with latent-variable modeling and makes the training objective look like a structured variational bound.
            """
        ),
        md(
            r"""
            The transition from GANs to diffusion is one of the most conceptually important turns in the course. GANs showed that excellent samples can arise from an implicit model trained by an adaptive critic, but they also exposed how difficult unstable game optimization can be. Diffusion responds by changing not only the objective, but the granularity of the generative task itself. Instead of asking for one direct leap from noise to image, it asks for many small denoising corrections. This change in decomposition is a large part of why diffusion training often feels much more robust in practice.
            """
        ),
        md(
            r"""
            There is a useful pedagogical contrast with the VAE chapter. A VAE introduced one latent variable and then spent most of its effort dealing with posterior intractability. A diffusion model takes the opposite path: it introduces a very large number of latent variables, one for each noise level in the chain, but chooses the forward process so carefully that many of the relevant Gaussian quantities become analytically tractable. The resulting model is therefore more elaborate in latent structure but, in some sense, simpler in local conditional algebra. This is one reason diffusion models often feel paradoxical at first: they are conceptually deeper than a vanilla autoencoder, yet some of their core training identities are easier to manipulate once the forward process has been designed.

            It is also helpful to say explicitly what problem diffusion is solving compared with GANs. GANs tried to learn a direct path from latent noise to realistic images through a learned adversarial game, and their difficulty lay largely in unstable optimization. Diffusion chooses a much longer route. It asks the model to solve many easy denoising problems instead of one hard synthesis problem. The price is sampling cost. The reward is a training objective that behaves much more like a supervised regression problem. Much of the modern success of diffusion models comes from this tradeoff.
            """
        ),
        md(
            r"""
            ## The Forward Noising Process

            Let $\boldsymbol{x}_0 \sim p_{gt}$ be a data sample. We define a forward Markov chain
            $$
            q(\boldsymbol{x}_{1:T} | \boldsymbol{x}_0)
            =
            \prod_{t=1}^T q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}),
            $$
            where each transition adds a small amount of Gaussian noise:
            $$
            q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1})
            =
            \mathcal{N}\big(
                \boldsymbol{x}_t;
                \sqrt{1-\beta_t}\,\boldsymbol{x}_{t-1},
                \beta_t \boldsymbol{I}
            \big).
            $$
            The sequence $\beta_t \in (0,1)$ is called the **noise schedule**. When $\beta_t$ is small, the transition only perturbs the image slightly. Repeating this many times gradually destroys structure until the terminal variable $\boldsymbol{x}_T$ is close to an isotropic Gaussian.

            It is helpful to introduce the shorthand $\alpha_t = 1-\beta_t$ and $\overline{\alpha}_t = \prod_{s=1}^t \alpha_s$. These quantities appear constantly because they summarize how much signal survives after many noising steps.
            """
        ),
        md(
            r"""
            ```{prf:theorem} Closed form of the forward marginal
            :label: thm-forward-marginal-ddpm

            For every $t \in \{1,\ldots,T\}$,
            $$
            q(\boldsymbol{x}_t | \boldsymbol{x}_0)
            =
            \mathcal{N}\big(
                \boldsymbol{x}_t;
                \sqrt{\overline{\alpha}_t}\,\boldsymbol{x}_0,
                (1-\overline{\alpha}_t)\boldsymbol{I}
            \big).
            $$
            Equivalently,
            $$
            \boldsymbol{x}_t
            =
            \sqrt{\overline{\alpha}_t}\,\boldsymbol{x}_0
            +
            \sqrt{1-\overline{\alpha}_t}\,\boldsymbol{\varepsilon},
            \qquad
            \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}).
            $$
            ```

            ```{prf:proof}
            We proceed by induction on $t$. For $t=1$ the formula is immediate because $\overline{\alpha}_1 = \alpha_1$ and
            $$
            q(\boldsymbol{x}_1 | \boldsymbol{x}_0)
            =
            \mathcal{N}\big(
                \boldsymbol{x}_1;
                \sqrt{\alpha_1}\,\boldsymbol{x}_0,
                (1-\alpha_1)\boldsymbol{I}
            \big).
            $$
            Assume the statement holds at time $t-1$, so that
            $$
            \boldsymbol{x}_{t-1}
            =
            \sqrt{\overline{\alpha}_{t-1}}\,\boldsymbol{x}_0
            +
            \sqrt{1-\overline{\alpha}_{t-1}}\,\boldsymbol{\varepsilon}_{t-1}
            $$
            for a standard Gaussian $\boldsymbol{\varepsilon}_{t-1}$. The next transition gives
            $$
            \boldsymbol{x}_t
            =
            \sqrt{\alpha_t}\,\boldsymbol{x}_{t-1}
            +
            \sqrt{1-\alpha_t}\,\boldsymbol{\eta}_t,
            \qquad
            \boldsymbol{\eta}_t \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}).
            $$
            Substituting the induction hypothesis,
            $$
            \boldsymbol{x}_t
            =
            \sqrt{\alpha_t\overline{\alpha}_{t-1}}\,\boldsymbol{x}_0
            +
            \sqrt{\alpha_t(1-\overline{\alpha}_{t-1})}\,\boldsymbol{\varepsilon}_{t-1}
            +
            \sqrt{1-\alpha_t}\,\boldsymbol{\eta}_t.
            $$
            The last two terms are independent centered Gaussians, so their sum is again Gaussian with covariance
            $$
            \alpha_t(1-\overline{\alpha}_{t-1})\boldsymbol{I}
            +
            (1-\alpha_t)\boldsymbol{I}
            =
            (1-\alpha_t\overline{\alpha}_{t-1})\boldsymbol{I}
            =
            (1-\overline{\alpha}_t)\boldsymbol{I}.
            $$
            Since $\alpha_t\overline{\alpha}_{t-1} = \overline{\alpha}_t$, the claimed formula follows.
            ```
            """
        ),
        md(
            r"""
            This theorem is one of the reasons diffusion models are computationally practical. It says that we do not need to simulate the entire forward chain to sample a noisy observation at time $t$. We can draw $\boldsymbol{x}_t$ directly from $\boldsymbol{x}_0$ in one step. During training, this allows us to select a random time index and generate the corresponding noisy sample without unrolling all earlier transitions.

            A concrete image example helps. If $\boldsymbol{x}_0$ is a clean handbag image and $t$ is small, the corresponding $\boldsymbol{x}_t$ still looks like the same handbag with a light dusting of Gaussian corruption. If $t$ is large, the shape is barely recognizable and the sample is close to pure static. Training teaches one network to operate across this whole range of difficulty.
            """
        ),
        md(
            r"""
            ```{admonition} Numerical Example: How Much Signal Survives?
            :class: numerical-example

            Take a toy noise schedule with $\beta_1 = 0.1$ and $\beta_2 = 0.2$. Then $\alpha_1 = 0.9$, $\alpha_2 = 0.8$, and $\overline{\alpha}_2 = 0.72$. The closed-form forward marginal says
            $$
            \boldsymbol{x}_2 = \sqrt{0.72}\,\boldsymbol{x}_0 + \sqrt{0.28}\,\boldsymbol{\varepsilon}.
            $$
            Numerically, $\sqrt{0.72} \approx 0.849$ and $\sqrt{0.28} \approx 0.529$. So after only two steps, about eighty-five percent of the original signal amplitude is still present, but a substantial noise term has already been mixed in.

            This is a useful sanity check for intuition. Early diffusion steps do not destroy the image completely. They weaken the signal gradually while increasing uncertainty, which is exactly why the reverse denoiser can hope to recover the structure one small step at a time.
            ```
            """
        ),
        md(
            r"""
            Conceptually, the theorem also explains why diffusion training can be randomized over time in such a simple way. The network does not need to see only neighboring states such as $\boldsymbol{x}_{t-1}$ and $\boldsymbol{x}_t$. Because $\boldsymbol{x}_t$ has a direct closed form given $\boldsymbol{x}_0$, the algorithm can choose a random noise level and present the model with the corresponding corrupted example immediately. This makes the training loop look much closer to ordinary supervised learning than one might expect from the long latent trajectory written in the model definition.

            For students coming from applied backgrounds, there is a useful interpretation here. The scalar $\overline{\alpha}_t$ measures how much signal survives by time $t$, while $1-\overline{\alpha}_t$ measures how much isotropic Gaussian corruption has been accumulated. Early times correspond to a slightly perturbed image. Late times correspond to something close to pure noise. The network is therefore exposed to a curriculum of denoising subproblems indexed by noise scale.
            """
        ),
        md(
            r"""
            ## The Reverse Model

            The forward chain is fixed and easy to sample from. The generative task is to learn a reverse chain
            $$
            p_\theta(\boldsymbol{x}_{0:T})
            =
            p(\boldsymbol{x}_T)\prod_{t=1}^T p_\theta(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t),
            $$
            where $p(\boldsymbol{x}_T)$ is chosen to be a standard Gaussian and each reverse transition is modeled as
            $$
            p_\theta(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t)
            =
            \mathcal{N}\big(
                \boldsymbol{x}_{t-1};
                \boldsymbol{\mu}_\theta(\boldsymbol{x}_t, t),
                \boldsymbol{\Sigma}_\theta(\boldsymbol{x}_t, t)
            \big).
            $$
            The neural network is therefore asked to undo one infinitesimal corruption step at a time. This local formulation is crucial. Directly generating a realistic image from white noise in one step is very hard. Predicting how to remove a little amount of Gaussian corruption from a partially structured image is far more manageable.
            """
        ),
        md(
            r"""
            One way to read this model is as a chain of microscopic decoders. Each reverse conditional takes a slightly too noisy image and tries to move it one step closer to the data manifold. No single step is responsible for creating the whole image. Global structure emerges cumulatively from the repeated composition of local denoising moves. This is very different from the generator in a GAN, which must learn a direct global transport from latent noise to the image space in one pass.

            This local perspective is also why U-Nets become such a natural architecture for diffusion. At intermediate times, the input to the network already contains a coarse version of the eventual image, buried under noise. The model does not need to invent all structure from scratch; it needs to preserve what is reliable, infer what is ambiguous, and progressively sharpen detail across scales. Architecturally, that is exactly the sort of task for which multi-scale skip-connected networks are well suited.
            """
        ),
        md(
            r"""
            ## The Posterior of One Forward Step

            To derive the training objective, we need the exact Gaussian posterior
            $$
            q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0).
            $$
            Intuitively, if we know both the clean image $\boldsymbol{x}_0$ and the noisy image $\boldsymbol{x}_t$, then the previous state $\boldsymbol{x}_{t-1}$ is no longer mysterious. It is just a Gaussian variable constrained by two linear Gaussian relations. The next theorem gives the exact formula.

            ```{prf:theorem} Closed form of the one-step posterior
            :label: thm-ddpm-posterior

            For $t \geq 2$,
            $$
            q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)
            =
            \mathcal{N}\big(
                \boldsymbol{x}_{t-1};
                \widetilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0),
                \widetilde{\beta}_t \boldsymbol{I}
            \big),
            $$
            where
            $$
            \widetilde{\beta}_t
            =
            \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_t}\beta_t
            $$
            and
            $$
            \widetilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0)
            =
            \frac{\sqrt{\overline{\alpha}_{t-1}}\beta_t}{1-\overline{\alpha}_t}\boldsymbol{x}_0
            +
            \frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t}\boldsymbol{x}_t.
            $$
            ```

            ```{prf:proof}
            By Bayes' rule and the Markov structure,
            $$
            q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)
            \propto
            q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1})q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_0).
            $$
            Both terms are Gaussian in $\boldsymbol{x}_{t-1}$. Writing their log densities and collecting quadratic and linear terms in $\boldsymbol{x}_{t-1}$ yields another Gaussian density. The precision matrix is scalar times the identity, so only scalar coefficients need to be combined. Completing the square gives posterior variance
            $$
            \widetilde{\beta}_t
            =
            \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_t}\beta_t
            $$
            and posterior mean
            $$
            \widetilde{\boldsymbol{\mu}}_t(\boldsymbol{x}_t, \boldsymbol{x}_0)
            =
            \frac{\sqrt{\overline{\alpha}_{t-1}}\beta_t}{1-\overline{\alpha}_t}\boldsymbol{x}_0
            +
            \frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline{\alpha}_t}\boldsymbol{x}_t.
            $$
            The calculation is straightforward but algebraically long, and its importance lies in the conclusion: the reverse posterior is Gaussian with parameters known in closed form once $\boldsymbol{x}_0$ is available.
            ```
            """
        ),
        md(
            r"""
            This posterior formula makes diffusion models look strikingly close to latent-variable models. The forward chain defines a tractable encoder-like distribution $q$. The reverse chain defines a learned decoder-like distribution $p_\theta$. The main difference from a VAE is that the latent structure is not a single vector but a long Markov chain whose conditionals have been chosen to admit closed-form Gaussian manipulations.
            """
        ),
        md(
            r"""
            It is worth stressing how unusual this design choice is. In a VAE, the variational distribution is learned because the exact posterior is too hard to compute. In DDPMs, much of the posterior structure is known analytically because the forward corruption was fixed in advance. The modeling freedom is moved away from the encoder side and concentrated in the reverse denoiser. This is one of the deepest structural differences between the two families, even though both are trained through an ELBO.

            The theorem also clarifies why the reverse model is asked to predict relatively simple Gaussian statistics rather than an arbitrary conditional density. Once the forward chain is chosen, the true reverse posterior conditioned on $\boldsymbol{x}_0$ is already Gaussian. The learning problem becomes: can a neural network infer the right Gaussian mean, or an equivalent target such as the added noise, from the noisy image and the time index alone?
            """
        ),
        md(
            r"""
            ## Variational Derivation of the DDPM Objective

            The log-likelihood of the model can be lower bounded using the forward process as a variational distribution. Starting from
            $$
            \log p_\theta(\boldsymbol{x}_0)
            =
            \log \int p_\theta(\boldsymbol{x}_{0:T})\, d\boldsymbol{x}_{1:T},
            $$
            we multiply and divide by $q(\boldsymbol{x}_{1:T} | \boldsymbol{x}_0)$ and apply Jensen's inequality exactly as in the VAE derivation. This yields the evidence lower bound
            $$
            \log p_\theta(\boldsymbol{x}_0)
            \geq
            \mathbb{E}_{q(\boldsymbol{x}_{1:T} | \boldsymbol{x}_0)}
            \left[
                \log p_\theta(\boldsymbol{x}_{0:T})
                -
                \log q(\boldsymbol{x}_{1:T} | \boldsymbol{x}_0)
            \right].
            $$
            After expanding the Markov factorizations and regrouping terms, one obtains a sum of Kullback-Leibler divergences between exact forward posteriors and learned reverse transitions, plus a terminal prior matching term and a reconstruction term at time $t=1$.
            """
        ),
        md(
            r"""
            ```{prf:theorem} ELBO decomposition for discrete diffusion
            :label: thm-ddpm-elbo

            Up to sign conventions, the negative ELBO can be decomposed as
            $$
            \mathcal{L}
            =
            \mathbb{E}_q\big[
                \operatorname{KL}(q(\boldsymbol{x}_T | \boldsymbol{x}_0) \| p(\boldsymbol{x}_T))
                +
                \sum_{t=2}^T
                \operatorname{KL}\big(
                    q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)
                    \|
                    p_\theta(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t)
                \big)
                - \log p_\theta(\boldsymbol{x}_0 | \boldsymbol{x}_1)
            \big].
            $$
            ```

            ```{prf:proof}
            Expand the ELBO integrand using
            $$
            p_\theta(\boldsymbol{x}_{0:T})
            =
            p(\boldsymbol{x}_T)\prod_{t=1}^T p_\theta(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t)
            $$
            and
            $$
            q(\boldsymbol{x}_{1:T} | \boldsymbol{x}_0)
            =
            \prod_{t=1}^T q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}).
            $$
            Rearranging the resulting sum and repeatedly inserting
            $$
            q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_t, \boldsymbol{x}_0)
            =
            \frac{q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1})q(\boldsymbol{x}_{t-1} | \boldsymbol{x}_0)}
                 {q(\boldsymbol{x}_t | \boldsymbol{x}_0)}
            $$
            produces one KL term for each reverse step, one KL term matching the terminal marginal to the chosen Gaussian prior, and one data reconstruction term involving $p_\theta(\boldsymbol{x}_0 | \boldsymbol{x}_1)$. The proof is structurally the same as in variational latent-variable models, except that the latent space is now the entire trajectory.
            ```
            """
        ),
        md(
            r"""
            The importance of this theorem is conceptual. A DDPM can be viewed as a very deep latent-variable model whose latent variables are the noisy intermediate states. The forward process plays the role of an analytically chosen inference model, while the reverse chain is learned. This is precisely why the connection with VAEs is not superficial. Both methods optimize a variational lower bound. They differ mainly in how the latent structure is designed and in how expressive the reverse conditionals become when the number of steps is large.
            """
        ),
        md(
            r"""
            This VAE connection deserves to be made even more explicit. If one writes the VAE ELBO schematically, one sees a reconstruction term plus a regularization term that compares an approximate posterior with a prior. In diffusion, the same broad logic survives, but it is distributed across time. Each KL term compares one exact local reverse posterior from the forward process with one learned local reverse conditional. Instead of asking one latent variable to summarize the whole image, the model asks a long chain of noisy latents to carry the information gradually. The gain is that each local denoising task is simple. The cost is that generation becomes iterative.

            From a course-design perspective, this is a valuable moment because it unifies two chapters that may otherwise feel unrelated. VAEs and DDPMs are not competing only at the level of sample quality. They represent two different choices about where to place tractability. VAEs keep sampling cheap but inference approximate. Diffusion keeps local posterior algebra easy but sampling long. Understanding this tradeoff helps students see why modern generative modeling is full of architectures that redistribute difficulty rather than eliminate it.
            """
        ),
        md(
            r"""
            ## Noise Prediction Parameterization

            The reverse mean can be parameterized in several equivalent ways. One especially successful choice is to make the neural network predict the noise $\boldsymbol{\varepsilon}$ used to generate $\boldsymbol{x}_t$ from $\boldsymbol{x}_0$. From the forward marginal identity,
            $$
            \boldsymbol{x}_t
            =
            \sqrt{\overline{\alpha}_t}\boldsymbol{x}_0
            +
            \sqrt{1-\overline{\alpha}_t}\boldsymbol{\varepsilon},
            $$
            we can solve for $\boldsymbol{x}_0$:
            $$
            \boldsymbol{x}_0
            =
            \frac{1}{\sqrt{\overline{\alpha}_t}}
            \left(
                \boldsymbol{x}_t
                -
                \sqrt{1-\overline{\alpha}_t}\,\boldsymbol{\varepsilon}
            \right).
            $$
            If a network $\boldsymbol{\varepsilon}_\theta(\boldsymbol{x}_t, t)$ predicts the noise, then this estimate can be substituted into the posterior mean formula. After fixing the reverse variance to a known schedule, the corresponding KL term reduces to a weighted mean squared error between the true noise and the predicted noise.
            """
        ),
        md(
            r"""
            In many practical expositions, the training objective is therefore written in the simplified form
            $$
            \mathcal{L}_{simple}(\theta)
            =
            \mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{\varepsilon}}
            \left[
                \|
                    \boldsymbol{\varepsilon}
                    -
                    \boldsymbol{\varepsilon}_\theta(\boldsymbol{x}_t, t)
                \|_2^2
            \right],
            $$
            where
            $$
            \boldsymbol{x}_t
            =
            \sqrt{\overline{\alpha}_t}\boldsymbol{x}_0
            +
            \sqrt{1-\overline{\alpha}_t}\boldsymbol{\varepsilon}.
            $$
            This is one of the most elegant features of DDPMs. A complicated variational objective turns into a denoising regression problem that is easy to implement and optimize. Later, when we discuss score-based models, we shall see that predicting the noise is closely related to predicting the score of the perturbed data distribution.
            """
        ),
        md(
            r"""
            The simplification above is not merely a coding convenience. It changes the way one should interpret the method. The network is no longer viewed primarily as a density model in the classical sense. It becomes a time-conditioned denoiser trained across many corruption scales. The probabilistic interpretation remains essential, because it explains why this denoising regression objective is legitimate, but the optimization face of the model is strikingly close to supervised learning on synthetic noisy targets.

            This is one reason diffusion models were comparatively easy to scale once the architectural ingredients became available. Deep learning already knows how to fit large regression networks well. If one can reformulate generative modeling so that the central neural task is "predict the noise that was added," a great deal of existing optimization machinery becomes immediately useful.
            """
        ),
        md(
            r"""
            ## DDIM and Deterministic Sampling

            The original DDPM sampler is stochastic because each reverse step samples from a Gaussian conditional. A natural question is whether one can keep the same training objective but sample along a more deterministic path. Denoising diffusion implicit models, or DDIMs, answer yes. They reinterpret the same learned denoiser through a non-Markovian construction whose marginals match the training family while permitting deterministic or partially stochastic reverse trajectories.

            The conceptual value of DDIM is very high. It shows that the learned network is not tied to one unique reverse chain. Once the model has learned how noisy observations relate to clean structure across time, one can choose different numerical trajectories for generation. Deterministic DDIM sampling often uses fewer steps than the original stochastic DDPM sampler while preserving much of the sample quality. This matters for practice because one of the main weaknesses of diffusion is slow generation.

            For students, DDIM is also the first strong hint that diffusion models are closely related to continuous transport viewpoints. The denoiser is trained from one probabilistic construction, yet can be used along another trajectory that looks much more like solving an evolution equation than repeatedly sampling fresh randomness. This observation will become even clearer in the continuous-time chapter when probability flow ODEs enter the picture.
            """
        ),
        md(
            r"""
            ## Relation with VAEs, Latent Diffusion, and Preview of the Continuous-Time Limit

            It is worth closing this discrete chapter by clarifying the relation with the models studied earlier. A VAE learns a decoder from a compact latent code and must approximate the posterior because exact inference is hard. A discrete diffusion model chooses a much more elaborate latent structure, but in exchange the forward corruption process is analytically fixed and its marginals and local posteriors are tractable. The result is an optimization problem that remains variational in spirit while being unusually stable in practice.

            Modern latent diffusion models add another twist to this story. Instead of running diffusion directly in pixel space, one first compresses images into a lower-dimensional latent representation using an autoencoding model, and then performs diffusion in that latent space. The motivation is computational rather than philosophical: denoising a compressed representation is dramatically cheaper than denoising full-resolution pixels, especially for high-resolution image synthesis. In this sense, latent diffusion can be viewed as a reconciliation between autoencoding ideas and diffusion ideas. The autoencoder provides a perceptually meaningful lower-dimensional space, and the diffusion model supplies the powerful generative prior inside that space.

            The price paid by diffusion models is therefore partly relocated rather than removed. Standard pixel-space diffusion is stable but slow. Latent diffusion makes sampling cheaper and scales better to large images, but it inherits the quality and inductive biases of the underlying latent representation. Understanding these tradeoffs is important because many practical text-to-image systems are best thought of not as raw DDPMs, but as carefully engineered latent diffusion pipelines.

            Finally, understanding what happens when the number of diffusion steps becomes very large leads naturally to continuous-time stochastic dynamics. That is the purpose of the next chapter, where the discrete denoising view will be reinterpreted through score matching, SDEs, and probability flow ODEs. The foundational DDPM perspective comes from {cite}`sohl2015deep` and {cite}`ho2020denoising`, while the deterministic sampling perspective and latent-space scaling directions are represented by {cite}`song2020ddim` and {cite}`rombach2022high`.
            """
        ),
        md(
            r"""
            ```{figure} ../assets/images/diffusion-diagram.png
            :width: 82%
            :align: center

            A conceptual picture of the diffusion process: data are gradually corrupted in the forward direction and progressively denoised in the learned reverse direction.
            ```
            """
        ),
    ],
)

write_notebook(
    "06-diffusion-models/score-based-modeling.ipynb",
    [
        md(
            r"""
            # Score-Based Models, SDEs, and Probability Flow ODEs

            The discrete diffusion chapter showed that image generation can be reformulated as the inversion of a long sequence of small Gaussian corruptions. The natural next question is what happens when the time step becomes very small and the number of steps becomes very large. In that regime, the noising process is better described as a stochastic differential equation, and the reverse dynamics become a continuous-time denoising flow. This perspective is not merely a mathematical curiosity. It clarifies why noise prediction, score prediction, stochastic sampling, and deterministic sampling are closely related rather than separate ideas.

            This chapter therefore develops the continuous-time viewpoint in a layered way. We first recall the intuitive difference between ordinary and stochastic differential equations. We then define score functions and explain why they are the right objects to learn along a family of perturbed data distributions. After that we introduce the forward SDE, the reverse-time SDE, and the probability flow ODE. Finally, we connect these ideas back to denoising score matching and to the VP and VE families that appear in modern diffusion models.
            """
        ),
        md(
            r"""
            For many students, this is the chapter where diffusion models stop looking like a clever discrete trick and start looking like a coherent probabilistic theory. The same model can now be described in at least three compatible languages: denoising regression, score estimation, and stochastic dynamics. None of these viewpoints is redundant. The denoising view makes optimization intuitive, the score view explains what function is actually being learned, and the SDE/ODE view explains how generation can proceed through either stochastic or deterministic evolution.

            It is worth slowing down here because continuous-time notation can create unnecessary anxiety. The aim of this chapter is not to turn the course into a full course on stochastic calculus. It is to make clear why continuous-time diffusion models are mathematically natural and why their practical algorithms are not disconnected hacks. Even students who do not master every analytic subtlety should leave with a clear picture of the moving pieces.
            """
        ),
        md(
            r"""
            ## An Intuitive Digression on ODEs and SDEs

            Before entering the score-based formalism, it is useful to pause for a conceptual distinction. An ordinary differential equation describes deterministic evolution. If a state $\boldsymbol{x}(t)$ follows
            $$
            \frac{d\boldsymbol{x}}{dt} = \boldsymbol{f}(\boldsymbol{x}(t), t),
            $$
            then the vector field $\boldsymbol{f}$ tells us the instantaneous velocity at every point and time. Once the initial condition is fixed, the trajectory is fixed as well. Deterministic transport models, including flow matching and continuous normalizing flows, live in this world.

            A stochastic differential equation adds a random perturbation at every instant. In informal notation,
            $$
            d\boldsymbol{x}
            =
            \boldsymbol{f}(\boldsymbol{x}, t)\,dt
            +
            g(t)\,d\boldsymbol{w},
            $$
            where $\boldsymbol{w}$ is a Brownian motion. The first term is still a drift that pushes the state in a coherent direction. The second term injects infinitesimal Gaussian randomness. As a result, the evolution of a single trajectory is random even when the initial condition is fixed, but the distribution of trajectories follows a well-defined law. This is precisely the level at which diffusion models operate: they are less concerned with one exact path than with how probability mass evolves over time.
            """
        ),
        md(
            r"""
            For students encountering this material for the first time, it helps to think of an ODE as a flow of particles in a velocity field and of an SDE as a flow of particles that are simultaneously advected and continuously shaken by noise. In image generation, the noising process is modeled by an SDE because we want the data distribution to spread gradually into a simpler reference distribution. The reverse generative process then learns how to undo this stochastic spreading.
            """
        ),
        md(
            r"""
            One more intuition can help. In an ODE, nearby particles that start at the same point and follow the same vector field remain synchronized because no external randomness separates them. In an SDE, even particles with the same initial condition can diverge because Brownian noise perturbs them differently. This is exactly why an SDE is such a good language for noising: it describes not just a deterministic deformation of the data manifold, but an actual spreading of probability mass into surrounding space.

            This distinction becomes practically important later when we compare stochastic reverse sampling with deterministic probability flow ODE sampling. The two procedures can share the same family of marginals even though one uses randomness and the other does not. Students often find this surprising at first, so it helps to keep the particle-flow picture in mind from the beginning.
            """
        ),
        md(
            r"""
            ## Scores and Perturbed Data Distributions

            Let $p_t$ denote a time-dependent probability density on image space. The **score** of this density is defined by
            $$
            \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x}).
            $$
            The score points in the direction of increasing log-density, so it can be interpreted as a local vector that tells us how the density rises around the point $\boldsymbol{x}$. Unlike the density itself, the score is invariant under unknown normalization constants, which is one reason it is attractive in high-dimensional modeling.

            In score-based generative modeling, the aim is not to learn only the score of the clean data distribution. Instead, one learns the scores of a whole family of perturbed distributions obtained by corrupting data with Gaussian noise of varying strength. This is crucial. The clean data distribution can be highly singular or concentrated near a thin manifold, making its score unstable or difficult to estimate directly. Once noise is added, the perturbed density becomes smoother and more regular, and the associated score becomes a better learning target.
            """
        ),
        md(
            r"""
            It is helpful to interpret the score geometrically. The vector $\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$ points toward regions where the current marginal density is higher. If a noisy sample lies in a low-density direction away from typical data, the score points back toward more likely structure. This is why score learning and denoising are so tightly linked. A good score field is, locally, a map of how probability mass should be pulled back toward realistic configurations.

            One can think of a noisy face image here. If random corruption pushes one eye slightly out of alignment or inserts grain in the cheek region, the score near that corrupted point points toward configurations that look more like typical faces. The score is therefore not an abstract derivative with no operational meaning. It is a directional hint about how to nudge a sample back toward the data distribution.

            There is also a normalization advantage here. Learning the density itself in high dimension is hard partly because the absolute scale of the density matters and normalizing constants are expensive. The score removes that obstacle by differentiating the log-density. One no longer needs the partition function explicitly. This is one of the main reasons score-based methods became so attractive theoretically.
            """
        ),
        md(
            r"""
            ## Denoising Score Matching

            Suppose we observe noisy samples of the form
            $$
            \widetilde{\boldsymbol{x}} = \boldsymbol{x} + \sigma \boldsymbol{\varepsilon},
            \qquad
            \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}),
            $$
            where $\boldsymbol{x} \sim p_{gt}$. Let $p_\sigma$ denote the density of the noisy variable $\widetilde{\boldsymbol{x}}$. A neural network $\boldsymbol{s}_\theta(\widetilde{\boldsymbol{x}}, \sigma)$ can be trained to approximate the score $\nabla_{\widetilde{\boldsymbol{x}}} \log p_\sigma(\widetilde{\boldsymbol{x}})$. One particularly useful objective is denoising score matching, which avoids explicit evaluation of the perturbed density.

            ```{prf:theorem} Optimal target in denoising score matching
            :label: thm-dsm-target

            Consider the objective
            $$
            \mathcal{L}_{DSM}(\theta)
            =
            \mathbb{E}_{\boldsymbol{x}, \widetilde{\boldsymbol{x}} | \boldsymbol{x}}
            \left[
                \left\|
                    \boldsymbol{s}_\theta(\widetilde{\boldsymbol{x}}, \sigma)
                    -
                    \nabla_{\widetilde{\boldsymbol{x}}}
                    \log q_\sigma(\widetilde{\boldsymbol{x}} | \boldsymbol{x})
                \right\|_2^2
            \right],
            $$
            where $q_\sigma(\widetilde{\boldsymbol{x}} | \boldsymbol{x}) = \mathcal{N}(\widetilde{\boldsymbol{x}}; \boldsymbol{x}, \sigma^2 \boldsymbol{I})$. The minimizer of this objective is
            $$
            \boldsymbol{s}_\theta^\star(\widetilde{\boldsymbol{x}}, \sigma)
            =
            \nabla_{\widetilde{\boldsymbol{x}}}\log p_\sigma(\widetilde{\boldsymbol{x}}).
            $$
            ```

            ```{prf:proof}
            For any fixed noisy observation $\widetilde{\boldsymbol{x}}$, the conditional expectation identity for squared error implies that the optimal predictor is
            $$
            \mathbb{E}
            \left[
                \nabla_{\widetilde{\boldsymbol{x}}}
                \log q_\sigma(\widetilde{\boldsymbol{x}} | \boldsymbol{x})
                \,|\,
                \widetilde{\boldsymbol{x}}
            \right].
            $$
            Since
            $$
            p_\sigma(\widetilde{\boldsymbol{x}})
            =
            \int q_\sigma(\widetilde{\boldsymbol{x}} | \boldsymbol{x}) p_{gt}(\boldsymbol{x})\, d\boldsymbol{x},
            $$
            differentiating under the integral sign gives
            $$
            \nabla_{\widetilde{\boldsymbol{x}}} p_\sigma(\widetilde{\boldsymbol{x}})
            =
            \int
            \nabla_{\widetilde{\boldsymbol{x}}} q_\sigma(\widetilde{\boldsymbol{x}} | \boldsymbol{x})
            p_{gt}(\boldsymbol{x})\, d\boldsymbol{x}.
            $$
            Dividing by $p_\sigma(\widetilde{\boldsymbol{x}})$ yields
            $$
            \nabla_{\widetilde{\boldsymbol{x}}}\log p_\sigma(\widetilde{\boldsymbol{x}})
            =
            \int
            \nabla_{\widetilde{\boldsymbol{x}}}
            \log q_\sigma(\widetilde{\boldsymbol{x}} | \boldsymbol{x})
            \, p(\boldsymbol{x} | \widetilde{\boldsymbol{x}})\, d\boldsymbol{x},
            $$
            which is exactly the conditional expectation above. Hence the optimal network equals the score of the perturbed density.
            ```
            """
        ),
        md(
            r"""
            Because the Gaussian conditional score is explicit,
            $$
            \nabla_{\widetilde{\boldsymbol{x}}}
            \log q_\sigma(\widetilde{\boldsymbol{x}} | \boldsymbol{x})
            =
            -\frac{\widetilde{\boldsymbol{x}} - \boldsymbol{x}}{\sigma^2},
            $$
            the objective becomes a tractable regression problem. This is the first major bridge between diffusion modeling and denoising: learning a score field can be turned into learning how corruption relates noisy samples back to clean ones.
            """
        ),
        md(
            r"""
            This theorem is one of the cleanest examples in the course of a hard generative objective being transformed into an easy regression target. The model is asked to predict a quantity derived from a Gaussian conditional that we know analytically, yet the minimizer recovers the score of the unknown noisy data distribution. In spirit, this mirrors what happened in DDPMs when the variational objective collapsed into noise prediction.

            A useful teaching point is that the network is never asked to estimate the score of the empirical data distribution directly at zero noise. Instead, it is trained across a continuum or grid of noise scales where the distributions are smoother. The difficult object is therefore approached through a family of easier auxiliary problems. This pattern is one of the recurring design ideas of modern generative modeling.
            """
        ),
        md(
            r"""
            ## Forward SDEs and Their Marginals

            In the continuous-time formulation, the noisy data process evolves according to an Itô SDE
            $$
            d\boldsymbol{x}
            =
            \boldsymbol{f}(\boldsymbol{x}, t)\,dt
            +
            g(t)\,d\boldsymbol{w},
            $$
            where $\boldsymbol{w}$ is standard Brownian motion. The drift $\boldsymbol{f}$ and diffusion amplitude $g$ are chosen so that, as time increases, the distribution of $\boldsymbol{x}(t)$ becomes progressively simpler. One usually wants the terminal law at time $T$ to be close to a standard Gaussian that is easy to sample from.

            Two particularly important families are the variance exploding and variance preserving SDEs. In the **VE-SDE**, the drift vanishes and noise strength grows over time:
            $$
            d\boldsymbol{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\, d\boldsymbol{w}.
            $$
            Here the signal itself is not contracted by the drift, but the variance of the perturbation keeps increasing. In the **VP-SDE**, one uses
            $$
            d\boldsymbol{x}
            =
            -\frac{1}{2}\beta(t)\boldsymbol{x}\,dt
            +
            \sqrt{\beta(t)}\, d\boldsymbol{w}.
            $$
            This process shrinks the signal while injecting noise, so the total variance remains controlled even though the initial information is gradually destroyed. For image generation, the VP family is especially important because it is the continuous-time analogue most closely connected to DDPMs.
            """
        ),
        md(
            r"""
            The VP versus VE distinction is worth making more concrete because the acronyms can otherwise feel like catalog labels. In a VP process, the signal amplitude is explicitly damped by the drift while noise is added, so one can think of the original image as being gradually forgotten and replaced by a controlled-variance noisy state. In a VE process, the original signal is not contracted by a drift term; instead, the observation is overwhelmed by increasingly large additive noise. Both routes can end near a simple reference law, but they organize the path to that law differently.

            This difference affects both theory and implementation. The natural parameterizations of time, the interpretation of the denoiser, and the numerical stiffness of the reverse dynamics can differ between VP and VE constructions. For the purposes of this course, the most important point is that DDPMs sit conceptually much closer to the VP family, whereas the broader score-based literature often discusses both VP and VE as parallel continuous-time noising choices.
            """
        ),
        md(
            r"""
            ## Reverse-Time SDE

            The forward SDE explains how to corrupt data. Generation requires the reverse evolution that starts from a simple terminal distribution and moves toward the data distribution. The remarkable fact is that the **reverse-time dynamics** are again governed by an SDE, but with a drift corrected by the score of the time-marginal density.

            ```{prf:theorem} Reverse-time SDE
            :label: thm-reverse-sde

            Let the forward process satisfy
            $$
            d\boldsymbol{x}
            =
            \boldsymbol{f}(\boldsymbol{x}, t)\,dt
            +
            g(t)\,d\boldsymbol{w},
            $$
            and let $p_t$ denote the density of $\boldsymbol{x}(t)$. Then, under suitable regularity assumptions, the reverse-time process satisfies
            $$
            d\boldsymbol{x}
            =
            \left[
                \boldsymbol{f}(\boldsymbol{x}, t)
                -
                g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})
            \right]dt
            +
            g(t)\,d\overline{\boldsymbol{w}},
            $$
            where time now runs backward and $\overline{\boldsymbol{w}}$ is a reverse-time Brownian motion.
            ```

            ```{prf:proof}
            We sketch the derivation at the level of densities and probability fluxes. The forward Itô SDE implies the Fokker-Planck equation
            $$
            \partial_t p_t(\boldsymbol{x})
            =
            -\nabla_{\boldsymbol{x}} \cdot
            \left(
                \boldsymbol{f}(\boldsymbol{x}, t)p_t(\boldsymbol{x})
            \right)
            +
            \frac{1}{2} g^2(t)\Delta p_t(\boldsymbol{x}).
            $$
            Since
            $$
            \Delta p_t
            =
            \nabla_{\boldsymbol{x}} \cdot
            \left(
                p_t \nabla_{\boldsymbol{x}} \log p_t
            \right),
            $$
            we can rewrite the diffusion term as
            $$
            \frac{1}{2} g^2(t)\Delta p_t
            =
            -\nabla_{\boldsymbol{x}} \cdot
            \left(
                -\frac{1}{2} g^2(t)
                p_t
                \nabla_{\boldsymbol{x}} \log p_t
            \right).
            $$
            Therefore the forward density evolution takes the continuity-equation form
            $$
            \partial_t p_t(\boldsymbol{x})
            =
            -\nabla_{\boldsymbol{x}} \cdot
            \left(
                p_t(\boldsymbol{x})
                \left[
                    \boldsymbol{f}(\boldsymbol{x}, t)
                    -
                    \frac{1}{2} g^2(t)
                    \nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x})
                \right]
            \right).
            $$

            Now reverse time by setting $s = T-t$ and requiring the reversed process to traverse the same family of marginals in the opposite order. The sign of the probability flux must therefore flip. If the reverse process has drift $\widetilde{\boldsymbol{f}}$, its own Fokker-Planck equation contributes not only the transport term $-\nabla \cdot (\widetilde{\boldsymbol{f}} p_t)$ but also a new diffusion term $+\frac{1}{2} g^2 \Delta p_t$. Matching the reversed flux exactly thus requires subtracting another $\frac{1}{2} g^2 \nabla \log p_t$ from the drift. Combining the two halves gives
            $$
            \widetilde{\boldsymbol{f}}(\boldsymbol{x}, t)
            =
            \boldsymbol{f}(\boldsymbol{x}, t)
            -
            g^2(t)\nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x}).
            $$
            Hence the reverse dynamics satisfy
            $$
            d\boldsymbol{x}
            =
            \left[
                \boldsymbol{f}(\boldsymbol{x}, t)
                -
                g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})
            \right]dt
            +
            g(t)\,d\overline{\boldsymbol{w}}.
            $$
            A fully rigorous proof requires time-reversal theory for diffusions, but this calculation captures the central mechanism: reversing a noising diffusion demands a score correction that steers mass back toward higher-density regions of the current marginal.
            ```
            """
        ),
        md(
            r"""
            This theorem contains the conceptual heart of score-based diffusion. The reverse drift is not arbitrary. It equals the forward drift corrected by a term that points toward regions of higher density in the current marginal distribution. In other words, the score tells the reverse dynamics how to denoise. The only missing ingredient is that the true score is unknown, so it must be approximated by a neural network $\boldsymbol{s}_\theta(\boldsymbol{x}, t)$ trained from noisy data.
            """
        ),
        md(
            r"""
            This is the place where the whole diffusion story condenses into one sentence: if you know the score of the time-marginal density at every noise level, then you know how to run the generative process backward. The unknown object in the reverse dynamics is therefore not an arbitrary conditional law but one geometrically meaningful vector field. This is why score estimation is not an auxiliary idea around diffusion; it is the central object that makes reverse sampling possible.

            From the viewpoint of scientific understanding, this theorem is also satisfying because it separates modeling from simulation. Modeling means estimating the score field. Simulation means numerically integrating either the reverse SDE or a related deterministic ODE once that field is available. The same trained network can therefore support multiple samplers, which is one reason the diffusion literature contains so much work on fast and improved generation methods.
            """
        ),
        md(
            r"""
            ## Tweedie's Formula

            Tweedie's formula makes the denoising interpretation even more explicit. It states that for Gaussian corruption, the posterior mean of the clean signal can be written directly in terms of the score of the noisy distribution.

            ```{prf:theorem} Tweedie's formula
            :label: thm-tweedie

            Let
            $$
            \widetilde{\boldsymbol{x}} = \boldsymbol{x} + \sigma \boldsymbol{\varepsilon},
            \qquad
            \boldsymbol{\varepsilon} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}),
            $$
            and let $p_\sigma(\widetilde{\boldsymbol{x}})$ denote the density of the noisy observation. Then
            $$
            \mathbb{E}[\boldsymbol{x} | \widetilde{\boldsymbol{x}}]
            =
            \widetilde{\boldsymbol{x}}
            +
            \sigma^2
            \nabla_{\widetilde{\boldsymbol{x}}}
            \log p_\sigma(\widetilde{\boldsymbol{x}}).
            $$
            ```

            ```{prf:proof}
            From the Gaussian conditional density,
            $$
            \nabla_{\widetilde{\boldsymbol{x}}}
            \log q_\sigma(\widetilde{\boldsymbol{x}} | \boldsymbol{x})
            =
            \frac{\boldsymbol{x} - \widetilde{\boldsymbol{x}}}{\sigma^2}.
            $$
            Taking the conditional expectation with respect to $\boldsymbol{x} | \widetilde{\boldsymbol{x}}$ and using the identity proved in the denoising score matching theorem gives
            $$
            \nabla_{\widetilde{\boldsymbol{x}}}
            \log p_\sigma(\widetilde{\boldsymbol{x}})
            =
            \frac{\mathbb{E}[\boldsymbol{x} | \widetilde{\boldsymbol{x}}] - \widetilde{\boldsymbol{x}}}{\sigma^2}.
            $$
            Rearranging yields the formula.
            ```
            """
        ),
        md(
            r"""
            Tweedie's formula shows that a score estimator is implicitly a denoiser. Once the score of the noisy distribution is known, one can recover the posterior mean of the clean sample. This is one reason diffusion, score matching, and denoising are so tightly intertwined in the literature. They are not separate intuitions accidentally leading to similar algorithms; they are different views of the same mathematical identity.
            """
        ),
        md(
            r"""
            This identity is pedagogically powerful because it translates an abstract differential quantity into an estimator of a familiar object. A student may wonder what the gradient of a log-density really buys us. Tweedie's formula answers: it tells us how to correct a noisy observation toward the posterior mean of the clean signal. In other words, the score is not just a geometric arrow field; it is directly actionable for denoising.

            One should also notice the continuity with the discrete diffusion chapter. There, the network predicted noise and from that prediction one could reconstruct an estimate of $\boldsymbol{x}_0$. Here, the score plays the analogous role in continuous time. The mathematical packages differ, but the operational meaning is the same: infer how the current noisy sample should move toward cleaner structure.
            """
        ),
        md(
            r"""
            ## Probability Flow ODE

            The reverse-time SDE is stochastic, but remarkably there exists a deterministic ODE with exactly the same time marginals. This is the **probability flow ODE**. It provides a bridge from stochastic diffusion models to deterministic transport models.

            ```{prf:theorem} Probability flow ODE
            :label: thm-probability-flow

            Consider the forward SDE
            $$
            d\boldsymbol{x}
            =
            \boldsymbol{f}(\boldsymbol{x}, t)\,dt
            +
            g(t)\,d\boldsymbol{w}.
            $$
            The deterministic ODE
            $$
            \frac{d\boldsymbol{x}}{dt}
            =
            \boldsymbol{f}(\boldsymbol{x}, t)
            -
            \frac{1}{2}g^2(t)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})
            $$
            has the same marginal densities $p_t$ as the forward SDE when evolved in the same time direction.
            ```

            ```{prf:proof}
            The proof follows by comparing the partial differential equations satisfied by the densities. The forward SDE induces a Fokker-Planck equation containing both drift and diffusion terms. One can rewrite the diffusion contribution as a divergence involving the score of the density. The ODE above is chosen so that its continuity equation produces exactly the same density evolution. The factor $\frac{1}{2}$ appears because the stochastic diffusion term contributes a second-order operator whose divergence form corresponds to half the square diffusion strength multiplied by the score field.
            ```
            """
        ),
        md(
            r"""
            This theorem is extremely important conceptually. It says that the same score network can support both stochastic sampling, through the reverse SDE, and deterministic sampling, through the probability flow ODE. In practice, the ODE viewpoint also enables likelihood-related calculations and creates a clear conceptual link with continuous normalizing flows and, later in the course, with flow matching.
            """
        ),
        md(
            r"""
            For many readers, this is the most surprising theorem in the chapter. One may expect that if the forward corruption used stochastic noise, then the reverse generative process must also be stochastic. The probability flow ODE says otherwise: there exists a deterministic evolution that shares the same one-time marginals as the stochastic reverse diffusion. This does not mean that individual trajectories match. It means that the distribution at each time is the same.

            This observation is one of the clearest conceptual bridges between diffusion and later flow matching ideas. A score-based diffusion model can be viewed not only as a stochastic denoiser but also as a learned transport system. Once students understand this, the transition to deterministic transport methods becomes far smoother.
            """
        ),
        md(
            r"""
            ## VP-SDE, VE-SDE, and the Relation with DDPMs

            We can now position the major continuous-time families more clearly. In the VP-SDE,
            $$
            d\boldsymbol{x}
            =
            -\frac{1}{2}\beta(t)\boldsymbol{x}\,dt
            +
            \sqrt{\beta(t)}\, d\boldsymbol{w},
            $$
            the mean signal decays exponentially while noise is injected continuously. This is the continuous analogue of the discrete Gaussian noising process used in DDPMs, and for this reason it is the main focus of these notes. In the VE-SDE, the drift is zero and the noise standard deviation grows directly with time. Both constructions produce families of perturbed densities whose scores can be learned, but they differ in numerical behavior and in the geometry of the perturbation path.

            From the discrete viewpoint, DDPM training often predicts the noise $\boldsymbol{\varepsilon}$. From the continuous-time viewpoint, the more intrinsic object is the score $\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$. The two parameterizations are equivalent up to analytic transformations determined by the perturbation law. This is why discrete diffusion, denoising score matching, and continuous-time score-based modeling form one coherent family rather than disconnected algorithms.
            """
        ),
        md(
            r"""
            This is also the right place to mention latent diffusion from the continuous-time viewpoint. Once score or noise prediction is understood as learning a reverse evolution in some representation space, there is no requirement that this space be the raw pixel grid. If an autoencoder provides a compressed latent representation in which semantics are preserved reasonably well, then the same VP- or VE-style reasoning can be applied there. The practical gain is enormous because the differential evolution is solved in a lower-dimensional and often smoother space. The conceptual cost is that one must now trust the latent representation as part of the generative model.
            """
        ),
        md(
            r"""
            ## Final Perspective for the Course

            At this point the continuous-time picture is in place. The forward process spreads data toward a simple reference distribution. The score network learns how density changes across that perturbation path. The reverse-time SDE converts this learned score into a stochastic sampler, while the probability flow ODE converts the same score into a deterministic sampler with identical marginals. Tweedie's formula explains why score estimation and denoising are two sides of the same coin.

            From the broader perspective of the course, diffusion is important not only because it is a successful model family, but because it teaches a very modern lesson about generative modeling. One can obtain powerful generators by designing a path between data and noise, learning the local geometry of that path, and then choosing an efficient reverse solver. This path-centric view will reappear in flow matching, where the stochastic path is replaced by a directly parameterized transport field.

            The next notebook will turn this theory into code. There we will implement sinusoidal time embeddings, a compact U-Net style denoiser, the noisy sample construction, and a basic sampling loop. The main references for this chapter are {cite}`song2021scorebased`, {cite}`song2020denoising`, {cite}`song2021maximum`, and {cite}`song2020ddim`.
            """
        ),
        md(
            r"""
            ```{figure} ../assets/images/sinusoidal_embedding.png
            :width: 72%
            :align: center

            Time-conditioning is a central ingredient in practical diffusion models, since the network must know which noise level or time point it is denoising.
            ```
            """
        ),
    ],
)

write_notebook(
    "06-diffusion-models/prompted-generation-and-modern-use.ipynb",
    [
        md(
            r"""
            # Prompted Generation, Conditional Diffusion, and Modern Use

            Up to this point, most of the course has described **unconditional generation**: sample noise, run a learned generative process, and obtain an image distributed like the training data. This is mathematically clean, but it does not yet explain the interface most people encounter in practice. Today, many public-facing systems are not used by pressing a button marked "sample one image from the prior." They are used by typing a prompt such as "a watercolor fox reading in a quiet library" and asking the model to generate an image that matches that instruction.

            This chapter therefore steps slightly away from derivations and takes a more **divulgative** perspective. The goal is to explain how modern diffusion-style image systems are typically used, what prompt-based generation changes relative to unconditional generation, and why the text interface feels so natural in contemporary AI products.
            """
        ),
        md(
            r"""
            ## Unconditional Versus Conditional Generation

            In an unconditional model, we aim to learn a distribution $p(\boldsymbol{x})$. Sampling produces an image with no external control beyond the randomness of the latent seed. This is useful for studying the core generative problem, but it offers limited user steering.

            In a conditional model, we instead learn or approximate a distribution such as $p(\boldsymbol{x} | \boldsymbol{c})$, where $\boldsymbol{c}$ is side information. The conditioning variable may be a class label, a segmentation mask, another image, or a text prompt. Once conditioning is available, sampling becomes guided rather than free-form. The user no longer asks only for "a realistic image"; the user asks for "a realistic image consistent with this description."
            """
        ),
        md(
            r"""
            A simple classroom example is the difference between sampling a random shoe from a trained fashion model and asking for "a red running shoe viewed from the side on a white background." The first task explores the marginal data distribution. The second explores a controlled slice of that distribution defined by a textual instruction. Modern image generation systems derive much of their usefulness from this second capability.
            """
        ),
        md(
            r"""
            ## Why Diffusion Became a Natural Home for Prompting

            Diffusion models are especially compatible with conditioning because the denoiser already receives contextual information such as time. Extending that interface to additional context is conceptually natural. Instead of learning only
            $$
            \boldsymbol{\varepsilon}_\theta(\boldsymbol{x}_t, t),
            $$
            one learns a conditioned denoiser such as
            $$
            \boldsymbol{\varepsilon}_\theta(\boldsymbol{x}_t, t, \boldsymbol{c}),
            $$
            where $\boldsymbol{c}$ may encode text or another control signal.

            The deep idea is that conditioning does not require inventing a completely different generative mechanism. It changes what information the denoiser is allowed to use while it performs each reverse step.
            """
        ),
        md(
            r"""
            In modern text-to-image systems, the prompt is first mapped into a representation by a language or text-embedding model. The diffusion denoiser then uses that representation while reversing noise. Architecturally, this is often implemented with **cross-attention**, meaning that image features at intermediate layers can attend to text features. The result is that the model does not merely know that "some text exists"; it can align different parts of the image-generation process with different semantic parts of the prompt.
            """
        ),
        md(
            r"""
            ## A High-Level Prompting Pipeline

            A contemporary prompt-based diffusion pipeline is often easiest to understand in the following stages:

            1. Encode the prompt into text features.
            2. Sample initial Gaussian noise in pixel space or, more commonly, in a learned latent space.
            3. Run the reverse denoising process while conditioning every step on the text representation.
            4. Decode the final latent, if a latent diffusion model is used, back into image space.

            The interface feels simple to the user, but internally the system is coordinating language understanding, image denoising, and often an autoencoding stage as well.
            """
        ),
        md(
            r"""
            It is useful to emphasize that the prompt does not act like a single command injected at the very start and then forgotten. In a conditioned diffusion model, the prompt influences the denoising trajectory repeatedly. One can think of the model as repeatedly asking: given the current noisy image, and given the text description, in which direction should this sample be denoised next?
            """
        ),
        md(
            r"""
            ## Classifier-Free Guidance in Intuition

            One of the most influential practical ideas is **classifier-free guidance**. The rough intuition is to train the model both with and without conditioning, then combine the two predictions during sampling. The unconditional branch says "move toward a plausible image." The conditional branch says "move toward an image matching the prompt." By amplifying the difference between them, one can increase prompt adherence.

            This usually improves instruction following, but not for free. Stronger guidance can reduce diversity or create over-sharpened, less natural outputs. This tradeoff is one reason why prompting quality is never only about model size. It also depends on how conditional information is injected and how sampling is guided.
            """
        ),
        md(
            r"""
            ```{admonition} Example: One Prompt, Two Modes of Use
            :class: numerical-example

            Consider the prompt: "a chalk drawing of a bicycle leaning against a cafe wall at sunset."

            In an unconditional generator trained on street scenes, there is no guarantee that sampling will ever produce this exact idea. It may generate a random street, a random bicycle, or no bicycle at all.

            In a prompted diffusion model, the text representation biases every reverse denoising step toward an image that matches *chalk drawing*, *bicycle*, *cafe wall*, and *sunset*. The final image is still stochastic because the initial noise and sampler matter, but the randomness now lives inside a semantic corridor defined by the prompt rather than across the full image distribution.
            ```
            """
        ),
        md(
            r"""
            ## Prompted Generation Versus Image Editing

            Prompting is not only for generation from scratch. The same conditional machinery can be used for **editing**, **inpainting**, **outpainting**, and **image-to-image translation**. In these settings, the condition is richer than pure text. It may include a source image, a mask indicating which pixels can change, or a structural guide such as edges or depth.

            This is one reason diffusion became so widely adopted in creative tools. The denoising process can be constrained in many ways, which makes it suitable not only for invention, but also for controlled revision.
            """
        ),
        md(
            r"""
            From a product perspective, this matters enormously. Many users do not want a random impressive picture. They want a specific poster refined, a background replaced, a product shot cleaned up, or a diagram restyled while preserving content. Prompted and conditioned diffusion systems are attractive precisely because they make these workflows possible within one shared probabilistic framework.
            """
        ),
        md(
            r"""
            ## How This Relates to Popular AI Assistants Today

            As of **May 22, 2026**, prompt-based image generation is publicly exposed in **ChatGPT** through its image-generation interface. In that product experience, the user can ask for a new image in natural language, refine the request conversationally, and edit existing images. At a high level, this is the public-facing form of conditional image generation: a language interface is used to specify the condition, and the image model produces outputs aligned with that condition.

            It is also useful to contrast this with **Claude**. Public Anthropic documentation currently describes Claude as supporting text output with image understanding and file creation workflows, but not as a general text-to-image chat system in the same way. This contrast is pedagogically helpful because it separates two ideas that are often conflated in public discussion: a multimodal assistant can *understand* images without necessarily being an image *generator* in the product sense.
            """
        ),
        md(
            r"""
            The important course takeaway is not product branding. It is the interface shift. Many people now encounter generative image models not by writing training code or sampling unconditional priors, but by chatting with a system and iteratively refining prompts. This means that, socially and educationally, the visible face of generative modeling has become strongly **conditional**, **interactive**, and **language-mediated**.
            """
        ),
        md(
            r"""
            ## Why Latent Diffusion Matters So Much in Practice

            Earlier we introduced latent diffusion mainly as a computational strategy. In the context of prompt-based systems, its importance becomes even clearer. High-resolution image generation is expensive in pixel space. If a good autoencoder provides a compressed latent representation, then denoising and conditioning can happen in that smaller space while still preserving semantics well enough for the final decoded image to look detailed.

            This design is one of the reasons modern prompt-based generation can be both expressive and computationally tractable. The language-conditioned reverse process happens where computation is cheaper, and the decoder reconstructs the final image afterward.
            """
        ),
        md(
            r"""
            ## A Healthy Public-Facing Interpretation

            From a divulgative point of view, one can summarize modern diffusion image systems in a sentence: they start from noise, repeatedly remove uncertainty, and use the prompt as a semantic guide during that cleanup. This sentence is not the full mathematics, but it is close enough to the truth to be educational without being misleading.

            At the same time, one should resist two oversimplifications. First, the prompt is not a magical spell that deterministically specifies one image; randomness and sampling design still matter. Second, the model is not "drawing like a human artist" in any direct psychological sense; it is following a learned high-dimensional denoising rule conditioned on text and data.
            """
        ),
        md(
            r"""
            ## Final Perspective

            The unconditional diffusion models studied in the previous chapters remain the cleanest route for understanding the mathematics. But the world students encounter today is largely one of *conditioned* generation. Prompting, editing, inpainting, and conversational refinement are now central to how the public experiences image synthesis.

            For that reason, a modern course should leave students fluent in both viewpoints. The unconditional model teaches the core probabilistic mechanism. The prompted model teaches how that mechanism becomes a usable interface for real systems.
            """
        ),
    ],
)

write_notebook(
    "06-diffusion-models/implementation.ipynb",
    [
        md(
            r"""
            # Diffusion Implementation Notebook

            This notebook turns the diffusion theory into a compact PyTorch implementation. As in the earlier implementation notebooks, the goal is clarity rather than scale. We keep the model small, the code explicit, and the training target tied closely to the theory. The implementation follows the standard DDPM idea: sample a clean image $\boldsymbol{x}_0$, choose a time index $t$, generate the corresponding noisy sample $\boldsymbol{x}_t$, and train a neural network to predict the Gaussian noise used in that corruption step.

            The most characteristic new ingredients are time conditioning and the denoising architecture. The network must know the noise level associated with each sample, so we encode the time index with sinusoidal embeddings. The denoiser itself is a compact U-Net style architecture with residual blocks, because the task is inherently multi-scale: the network must combine local image detail with global context when estimating which part of the signal is structure and which part is noise.
            """
        ),
        md(
            r"""
            For classroom use, it is helpful to state expectations clearly. This notebook is not meant to compete with industrial diffusion systems. It is meant to expose the core algorithmic skeleton in a form that can be read line by line. Students should therefore pay attention not only to what each function computes, but also to which theoretical object it corresponds to: the forward marginal, the time embedding, the denoiser, the noise-prediction loss, and the reverse sampler.

            It is also useful to keep a hierarchy of realism in mind. At the mathematical level, the notebook implements the DDPM logic faithfully. At the architectural level, it uses a very small U-Net compared with real image-generation systems. At the data level, `FashionMNIST` is chosen because it lets the whole pipeline run in a compact form. The lesson is therefore structural rather than performance-oriented.
            """
        ),
        md(
            r"""
            ## Imports and Dataset

            We again use `FashionMNIST` to keep the training loop lightweight. The images are linearly rescaled from $[0,1]$ to $[-1,1]$, which is common in diffusion implementations because the corrupted and reconstructed tensors are then centered more naturally around zero.
            """
        ),
        md(
            r"""
            The hyperparameters are intentionally moderate but no longer merely toy-sized. Three hundred diffusion steps are enough to show the iterative nature of sampling while giving the denoiser a useful noise curriculum. The base channel count is still small by research standards, but large enough for the samples to be recognizable after training.
            """
        ),
        code(
            """
            import math
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from pathlib import Path
            from torch.utils.data import DataLoader
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
            from torchvision import datasets, transforms, utils
            from tqdm.auto import tqdm
            import matplotlib.pyplot as plt

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(7)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(7)
            num_workers = 2 if device.type == "cuda" else 0
            project_root = Path.cwd() if (Path.cwd() / "_config.yml").exists() else Path.cwd().parent
            DATA_ROOT = project_root / "data"

            batch_size = 128
            image_size = 28
            channels = 1
            base_channels = 64
            time_dim = 128
            timesteps = 300
            epochs = 30
            lr = 2e-4

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ])

            train_dataset = datasets.FashionMNIST(
                root=DATA_ROOT,
                train=True,
                download=True,
                transform=transform,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )
            """
        ),
        md(
            r"""
            ## Noise Schedule and Forward Process

            We discretize the variance preserving process through a sequence of $\beta_t$ values. The coefficients $\alpha_t = 1-\beta_t$ and $\overline{\alpha}_t = \prod_{s=1}^t \alpha_s$ are precomputed once and then reused throughout training and sampling. The function `q_sample` implements the closed-form forward marginal
            $$
            \boldsymbol{x}_t
            =
            \sqrt{\overline{\alpha}_t}\,\boldsymbol{x}_0
            +
            \sqrt{1-\overline{\alpha}_t}\,\boldsymbol{\varepsilon},
            $$
            which is exactly the identity derived in the discrete diffusion theory notebook.
            """
        ),
        md(
            r"""
            This is one of the best places to connect code back to theory explicitly. The arrays `betas`, `alphas`, and `alpha_bars` are not arbitrary implementation details. They are the discretized coefficients that determine how much signal survives and how much noise accumulates at each time. The helper function `extract` exists only because different items in a minibatch may have different sampled times, so the scalar schedule coefficients must be reshaped to broadcast over image tensors correctly.

            Students sometimes underestimate how important this direct-sampling identity is. Without `q_sample`, training would require simulating every intermediate step from $\boldsymbol{x}_0$ up to the sampled time index. With it, each minibatch can jump immediately to the desired noise level. This is part of why diffusion training is far cheaper than a naive reading of the long latent chain might suggest.
            """
        ),
        code(
            """
            betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
            alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]], dim=0)

            sqrt_alpha_bars = torch.sqrt(alpha_bars)
            sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
            sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
            posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)


            def extract(coefficients, t, x_shape):
                out = coefficients.gather(0, t)
                return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))


            def q_sample(x0, t, noise=None):
                if noise is None:
                    noise = torch.randn_like(x0)
                return (
                    extract(sqrt_alpha_bars, t, x0.shape) * x0
                    + extract(sqrt_one_minus_alpha_bars, t, x0.shape) * noise
                )
            """
        ),
        md(
            r"""
            ## Sinusoidal Time Embeddings

            The denoiser receives not only an image-like tensor, but also a time index. A simple integer index is not an especially expressive representation, so we map time to a vector using sinusoidal features. This mirrors the positional encoding idea used in transformers and has become a standard way to let diffusion models represent the notion of noise level smoothly across time.
            """
        ),
        md(
            r"""
            From a modeling point of view, this time embedding is what turns one network into a whole family of denoisers. The model should behave differently when the input is almost clean than when it is nearly pure noise. A raw scalar time index does contain that information, but the sinusoidal map gives the downstream layers a richer basis in which to represent time-dependent behavior smoothly. This is why time conditioning is not an optional detail in diffusion models. It is part of the core problem specification.
            """
        ),
        code(
            """
            class SinusoidalTimeEmbedding(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.dim = dim

                def forward(self, t):
                    half_dim = self.dim // 2
                    factor = math.log(10000.0) / max(half_dim - 1, 1)
                    frequencies = torch.exp(
                        torch.arange(half_dim, device=t.device) * -factor
                    )
                    angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
                    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
                    if self.dim % 2 == 1:
                        embedding = F.pad(embedding, (0, 1))
                    return embedding
            """
        ),
        md(
            r"""
            ## A Compact Residual U-Net

            The architecture below is intentionally modest, but it still captures the core structural ideas used in practical diffusion models. Residual blocks process features while injecting the time embedding. Downsampling creates a coarser representation with a larger receptive field. Upsampling then reconstructs detail while using skip connections to preserve spatial information. This is the same general U-Net logic discussed in the early neural-network chapter, now specialized to denoising across noise scales.
            """
        ),
        md(
            r"""
            The most important architectural lesson is not the exact number of layers. It is the pattern of information flow. The network first builds local features, then aggregates coarser global context, and finally reconstructs high-resolution predictions with skip-connected access to earlier detail. This is exactly the kind of computation a denoiser needs: preserve reliable local structure, reason globally about ambiguous shapes, and then return to pixel space with both forms of information available.

            In larger systems, one often adds attention blocks, deeper hierarchies, more channels, normalization refinements, or classifier-free conditioning. But those elaborations extend the same logic already visible here. For teaching purposes, the small U-Net is valuable because every component still fits into one notebook without losing the conceptual backbone of modern diffusion models.
            """
        ),
        code(
            """
            class ResidualBlock(nn.Module):
                def __init__(self, in_channels, out_channels, time_dim):
                    super().__init__()
                    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                    self.time_mlp = nn.Linear(time_dim, out_channels)
                    self.norm1 = nn.GroupNorm(8, out_channels)
                    self.norm2 = nn.GroupNorm(8, out_channels)
                    self.activation = nn.SiLU()
                    self.residual = (
                        nn.Conv2d(in_channels, out_channels, kernel_size=1)
                        if in_channels != out_channels else nn.Identity()
                    )

                def forward(self, x, t_emb):
                    h = self.conv1(x)
                    h = self.norm1(h)
                    h = self.activation(h)

                    # Broadcast the time embedding across spatial positions.
                    time_term = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
                    h = h + time_term

                    h = self.conv2(h)
                    h = self.norm2(h)
                    h = self.activation(h)
                    return h + self.residual(x)


            class SmallUNet(nn.Module):
                def __init__(self, in_channels=1, base_channels=64, time_dim=128):
                    super().__init__()
                    self.time_embedding = nn.Sequential(
                        SinusoidalTimeEmbedding(time_dim),
                        nn.Linear(time_dim, time_dim),
                        nn.SiLU(),
                        nn.Linear(time_dim, time_dim),
                    )

                    self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

                    self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
                    self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
                    self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
                    self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
                    self.down3 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

                    self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

                    self.upsample2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
                    self.up2 = ResidualBlock(base_channels * 4, base_channels * 2, time_dim)
                    self.upsample1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
                    self.up1 = ResidualBlock(base_channels * 2, base_channels, time_dim)
                    self.output_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

                def forward(self, x, t):
                    t_emb = self.time_embedding(t)

                    # Save an early feature map so the decoder can recover detail later.
                    x0 = self.input_conv(x)
                    x1 = self.down1(x0, t_emb)
                    x2 = self.downsample1(x1)
                    x2 = self.down2(x2, t_emb)
                    x3 = self.downsample2(x2)
                    x3 = self.down3(x3, t_emb)

                    x_mid = self.mid(x3, t_emb)

                    x_up = self.upsample2(x_mid)
                    x_up = torch.cat([x_up, x2], dim=1)
                    x_up = self.up2(x_up, t_emb)
                    x_up = self.upsample1(x_up)
                    # Skip connection joins coarse semantics with finer spatial detail.
                    x_up = torch.cat([x_up, x1], dim=1)
                    x_up = self.up1(x_up, t_emb)
                    return self.output_conv(x_up)


            model = SmallUNet(in_channels=channels, base_channels=base_channels, time_dim=time_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            """
        ),
        md(
            r"""
            ## Noise Prediction Objective

            In this implementation the network predicts the noise tensor directly. This corresponds to the simplified DDPM training objective discussed in the discrete diffusion chapter:
            $$
            \mathbb{E}\big[\|\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\boldsymbol{x}_t, t)\|_2^2\big].
            $$
            The elegance of the method lies in the fact that a complex variational derivation reduces to an ordinary regression loss between the sampled corruption noise and the network output.
            """
        ),
        md(
            r"""
            This is an excellent point to remind students what the network is *not* doing. It is not directly predicting a clean image, and it is not scoring realism with an adversarial critic. It is learning the noise component of a corrupted sample. From that apparently modest target, the rest of the generative machinery follows. This is one of the central philosophical moves of diffusion modeling: solve a simpler local inverse problem many times, rather than a single difficult global synthesis problem.
            """
        ),
        code(
            """
            def diffusion_loss(model, x0, t):
                # Sample the corruption explicitly so the target noise is known.
                noise = torch.randn_like(x0)
                xt = q_sample(x0, t, noise)
                predicted_noise = model(xt, t)
                return F.mse_loss(predicted_noise, noise)
            """
        ),
        md(
            r"""
            ## Training Loop

            The loop is structurally simple. Each minibatch produces a fresh random time index for every image, because we want the denoiser to learn all noise levels rather than memorize one fixed step. This randomization is one of the most characteristic aspects of diffusion training.
            """
        ),
        md(
            r"""
            The line that samples `t` uniformly at random is doing substantial conceptual work. It means the model is trained across the whole denoising curriculum simultaneously. Some samples in a batch are almost clean and ask for small corrections. Others are heavily corrupted and ask for coarse structural recovery. The shared network must learn to organize these regimes using the time embedding as context.

            In larger research codebases, training often includes mixed precision, exponential moving averages, richer schedules over time sampling, and distributed data loading. None of those changes the basic logic presented here. The pedagogical aim of this notebook is to make that logic visible before engineering complexity is added.
            """
        ),
        code(
            """
            history = []

            for epoch in tqdm(range(epochs), desc="Diffusion epochs"):
                model.train()
                running_loss = 0.0

                for x0, _ in tqdm(train_loader, desc="train", leave=False):
                    x0 = x0.to(device)
                    # Every image is trained at a randomly chosen noise level.
                    t = torch.randint(0, timesteps, (x0.size(0),), device=device).long()

                    optimizer.zero_grad()
                    loss = diffusion_loss(model, x0, t)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / len(train_loader)
                history.append(epoch_loss)
                print(f"Epoch {epoch + 1:02d} | loss: {epoch_loss:.6f}")
            """
        ),
        md(
            r"""
            The scale of the loss is usually much easier to interpret here than in GAN training, because we are again solving a straightforward regression problem. Even so, visual inspection remains important. A lower MSE does not automatically guarantee visually good long-chain sampling if the model has not learned to denoise well across all time scales.
            """
        ),
        md(
            r"""
            A useful rule of thumb for students is that diffusion losses are trustworthy but incomplete. If the loss decreases steadily, the model is almost certainly learning something meaningful. But the generated images may still remain blurry, unstable across late reverse steps, or semantically weak if the architecture is too small. This makes diffusion easier to monitor than GANs, but not trivial to judge from a scalar alone.
            """
        ),
        md(
            r"""
            ## Reverse Sampling

            To generate images, we start from Gaussian noise and iterate the learned reverse transition. The function below uses the standard DDPM ancestral update with a predicted noise term. For simplicity, we fix the reverse variance to the closed-form posterior variance schedule. This is enough to produce a fully end-to-end pedagogical sampler.
            """
        ),
        md(
            r"""
            This sampler is the place where the notebook becomes unmistakably generative. Training only teaches the model to predict noise at random times. Sampling composes those local denoising moves into a full trajectory from Gaussian noise back to image space. Students should notice that this is also where diffusion pays its computational price: one forward pass through the denoiser is not enough. Generation requires many such passes, one for each reverse step.

            This is exactly why ideas such as DDIM, ODE samplers, and latent diffusion matter so much in practice. They do not replace the denoiser. They change the trajectory or the space in which the trajectory is followed, with the goal of reducing this iterative cost while preserving sample quality.
            """
        ),
        code(
            """
            @torch.no_grad()
            def p_sample(model, x, t_scalar):
                # Build a batch-shaped time tensor for the current reverse step.
                t = torch.full((x.size(0),), t_scalar, device=device, dtype=torch.long)
                beta_t = extract(betas, t, x.shape)
                sqrt_one_minus_alpha_bar_t = extract(sqrt_one_minus_alpha_bars, t, x.shape)
                sqrt_recip_alpha_t = extract(sqrt_recip_alphas, t, x.shape)

                predicted_noise = model(x, t)
                model_mean = sqrt_recip_alpha_t * (
                    x - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t
                )

                if t_scalar == 0:
                    return model_mean

                # Add the stochastic part of the reverse transition except at the last step.
                variance = extract(posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                return model_mean + torch.sqrt(variance) * noise


            @torch.no_grad()
            def sample(model, n_samples=16, show_progress=True):
                model.eval()
                x = torch.randn(n_samples, channels, image_size, image_size, device=device)

                # Reverse the whole noising chain one step at a time.
                reverse_steps = reversed(range(timesteps))
                if show_progress:
                    reverse_steps = tqdm(reverse_steps, total=timesteps, desc="sampling", leave=False)
                for t_scalar in reverse_steps:
                    x = p_sample(model, x, t_scalar)

                x = x.clamp(-1, 1)
                x = 0.5 * (x + 1.0)
                return x.cpu()
            """
        ),
        code(
            """
            samples = sample(model, n_samples=16)
            image = utils.make_grid(samples, nrow=4, pad_value=1.0)
            plt.figure(figsize=(6, 6))
            plt.imshow(image.permute(1, 2, 0), cmap="gray")
            plt.axis("off")
            plt.show()
            """
        ),
        code(
            """
            plt.figure(figsize=(7, 4))
            plt.plot(history)
            plt.xlabel("Epoch")
            plt.ylabel("Noise prediction loss")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## Quantitative Evaluation with FID and KID

            Diffusion models are usually judged with the same distribution-level image metrics used for GANs and VAEs. In practice, one often reports FID and sometimes KID on a held-out reference set. The code below follows the same `torchmetrics` pattern used earlier, with one additional reminder: because our diffusion samples already return values in $[0,1]$, the preprocessing mainly consists of turning grayscale images into three channels.
            """
        ),
        code(
            """
            def prepare_for_inception_metrics(images):
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                return images.clamp(0.0, 1.0)


            @torch.no_grad()
            def compute_diffusion_fid_and_kid(model, real_loader, device, num_fake=1000):
                fid = FrechetInceptionDistance(
                    feature=2048,
                    normalize=True,
                    reset_real_features=False,
                ).set_dtype(torch.float64).to(device)
                kid = KernelInceptionDistance(
                    feature=2048,
                    subsets=10,
                    subset_size=100,
                    normalize=True,
                    reset_real_features=False,
                ).to(device)

                for real_images, _ in tqdm(real_loader, desc="real metrics", leave=False):
                    real_images = prepare_for_inception_metrics(0.5 * (real_images.to(device) + 1.0))
                    fid.update(real_images, real=True)
                    kid.update(real_images, real=True)

                generated = 0
                pbar = tqdm(total=num_fake, desc="diffusion fake metrics", leave=False)
                while generated < num_fake:
                    batch_n = min(batch_size, num_fake - generated)
                    fake_images = sample(model, n_samples=batch_n, show_progress=False).to(device)
                    fake_images = prepare_for_inception_metrics(fake_images)
                    fid.update(fake_images, real=False)
                    kid.update(fake_images, real=False)
                    generated += batch_n
                    pbar.update(batch_n)
                pbar.close()

                kid_mean, kid_std = kid.compute()
                return {
                    "fid": fid.compute().item(),
                    "kid_mean": kid_mean.item(),
                    "kid_std": kid_std.item(),
                }


            metric_scores = compute_diffusion_fid_and_kid(model, train_loader, device)
            print(metric_scores)
            """
        ),
        md(
            r"""
            For diffusion, these scores often line up better with the visual experience than they do for tiny VAEs, because diffusion models tend to cover the data distribution more broadly while preserving decent sample quality. Even so, one should remain careful. A model can improve its FID while still sampling too slowly for a practical application, and no scalar metric captures the full tradeoff among quality, diversity, controllability, and compute.
            """
        ),
        md(
            r"""
            ## Practical Discussion

            Even in this compact notebook, the most important ingredients of a modern diffusion model are already visible: a noise schedule, time conditioning, a U-Net style denoiser, the noise-prediction loss, and an iterative reverse sampler. Larger models refine each of these pieces rather than replacing them completely. They use stronger U-Nets, better schedules, more sophisticated parameterizations, and sometimes faster samplers such as DDIM or ODE-based methods, but the skeleton remains the same.

            Pedagogically, this notebook is especially valuable because it closes the loop opened by the theory chapters. The discrete diffusion chapter explained why the noise-prediction objective arises from a variational decomposition. The continuous-time chapter explained why denoising, scores, and reverse stochastic dynamics are mathematically aligned. Here all of that becomes executable code.
            """
        ),
        md(
            r"""
            When this notebook behaves reasonably well, one should expect the reverse samples to become progressively more recognizable as training proceeds, even if they remain far from the fidelity of modern large-scale systems. If the outputs remain pure noise, inspect whether the loss is decreasing and whether time conditioning is wired correctly. If outputs collapse to nearly identical shapes, the model capacity may be too small or the number of epochs too limited. If training is numerically unstable, the usual suspects are learning rate, normalization, or shape mismatches in the schedule broadcasting.

            The broader lesson is that diffusion models are unusually clean examples of theory and implementation reinforcing each other. The theoretical derivation is not decorative; it tells us exactly what target to regress, why time conditioning is necessary, and why reverse sampling must be iterative. This makes the notebook especially valuable for a PhD course, where students should be able to map equations to code without hidden conceptual jumps.
            """
        ),
    ],
)

write_notebook(
    "07-flow-matching/flow-matching.ipynb",
    [
        md(
            r"""
            # Flow Matching

            **Flow matching** can be understood as the deterministic sibling of continuous-time diffusion modeling. In the diffusion chapter, a stochastic differential equation transported a simple distribution toward the data distribution, and the same marginals could also be realized through a probability flow ODE. Flow matching starts directly from that deterministic viewpoint. Instead of first defining a noisy stochastic process and then extracting an associated ODE, one defines a time-dependent probability path between an easy source distribution and the target data distribution, then learns a velocity field whose flow realizes that path.

            This perspective is attractive for several reasons. It avoids explicit simulation of stochastic noise during sampling, it connects naturally with continuous normalizing flows, and it separates the design of the probability path from the learning of the transport dynamics. The central question is no longer how to reverse a noising process, but how to choose and learn a vector field that pushes mass in the right way over time.
            """
        ),
        md(
            r"""
            For course purposes, flow matching is important because it reveals that the diffusion story was never only about denoising Gaussian noise. It was also about learning a path between a simple distribution and the data distribution. Flow matching keeps that path-based perspective but removes the stochastic wrapper. This makes the chapter a natural culmination of several earlier ideas: latent geometry from VAEs, transport thinking from diffusion, and regression-style training objectives that turn generative modeling into a learnable supervised problem.

            The chapter is also pedagogically useful because it forces us to separate two questions that are often conflated. The first question is *which family of distributions should connect source and target over time?* The second is *which velocity field realizes that family?* Flow matching treats the first as a design choice and the second as a learning problem. That separation is one of the most conceptually elegant features of the method.
            """
        ),
        md(
            r"""
            ## Deterministic Transport and the Continuity Equation

            Let $\boldsymbol{x}(t)$ be a deterministic trajectory solving the ODE $\frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}(\boldsymbol{x}(t), t)$, where $\boldsymbol{v}$ is a time-dependent velocity field. If the initial condition $\boldsymbol{x}(0)$ is random with density $p_0$, then the density $p_t$ of $\boldsymbol{x}(t)$ evolves according to a conservation law. Intuitively, probability mass does not disappear. It is merely redistributed by the flow. This conservation principle leads to the continuity equation
            $$
            \partial_t p_t(\boldsymbol{x})
            +
            \nabla_{\boldsymbol{x}} \cdot \big(p_t(\boldsymbol{x}) \boldsymbol{v}(\boldsymbol{x}, t)\big)
            = 0.
            $$
            The equation says that temporal change in density is balanced by the divergence of probability flux.
            """
        ),
        md(
            r"""
            ```{prf:theorem} Continuity equation for deterministic flows
            :label: thm-continuity-equation

            Suppose $\boldsymbol{x}(t)$ solves the ODE
            $$
            \frac{d\boldsymbol{x}}{dt} = \boldsymbol{v}(\boldsymbol{x}(t), t)
            $$
            and let $p_t$ denote the density of $\boldsymbol{x}(t)$. Under standard regularity assumptions, the family $(p_t)_{t \in [0,1]}$ satisfies
            $$
            \partial_t p_t(\boldsymbol{x})
            +
            \nabla_{\boldsymbol{x}} \cdot \big(p_t(\boldsymbol{x}) \boldsymbol{v}(\boldsymbol{x}, t)\big)
            = 0.
            $$
            ```

            ```{prf:proof}
            A full proof may be given in weak form. Let $\varphi$ be a smooth compactly supported test function. Then
            $$
            \frac{d}{dt}\mathbb{E}[\varphi(\boldsymbol{x}(t))]
            =
            \mathbb{E}\left[
                \nabla \varphi(\boldsymbol{x}(t))^\top
                \boldsymbol{v}(\boldsymbol{x}(t), t)
            \right]
            $$
            by the chain rule. Writing expectations as integrals against $p_t$ gives
            $$
            \frac{d}{dt}\int \varphi(\boldsymbol{x}) p_t(\boldsymbol{x})\, d\boldsymbol{x}
            =
            \int \nabla \varphi(\boldsymbol{x})^\top
            \boldsymbol{v}(\boldsymbol{x}, t)
            p_t(\boldsymbol{x})\, d\boldsymbol{x}.
            $$
            Integrating by parts moves the gradient from $\varphi$ onto the probability flux, yielding
            $$
            \int \varphi(\boldsymbol{x})
            \left[
                \partial_t p_t(\boldsymbol{x})
                +
                \nabla_{\boldsymbol{x}} \cdot
                \big(p_t(\boldsymbol{x}) \boldsymbol{v}(\boldsymbol{x}, t)\big)
            \right]
            d\boldsymbol{x}
            = 0.
            $$
            Since this holds for every test function $\varphi$, the quantity in brackets must vanish, which proves the claim.
            ```
            """
        ),
        md(
            r"""
            This equation is the deterministic counterpart of the Fokker-Planck equation studied implicitly in the diffusion chapter. There, density evolution involved both drift and stochastic diffusion. Here only transport remains. Flow matching is therefore a method for learning transport fields rather than score-corrected stochastic dynamics.
            """
        ),
        md(
            r"""
            It is helpful to interpret the continuity equation physically. Imagine colored dye being moved by a fluid velocity field. The dye concentration changes at each location not because mass is created or destroyed, but because the fluid carries it elsewhere. Probability density behaves in the same way under deterministic flow. This picture is often more intuitive than the partial differential equation itself, and it clarifies why vector-field design is central.

            The analogy also helps explain why deterministic transport can be attractive computationally. If one can learn a vector field that moves mass coherently along a good path, one may avoid some of the random fluctuation handling that appears in diffusion-based sampling. The whole burden then shifts to choosing a path and matching the right transport velocity.
            """
        ),
        md(
            r"""
            ## Probability Paths

            The next modeling ingredient is a probability path $(p_t)_{t \in [0,1]}$ such that $p_0$ is easy to sample and $p_1 = p_{gt}$ is the target data distribution. The source $p_0$ is often chosen to be a standard Gaussian. The target is the data distribution. Once a path is fixed, one wants a vector field $\boldsymbol{v}(\boldsymbol{x}, t)$ such that the continuity equation transports $p_0$ along that path and reaches $p_1$ at time one.

            At first sight, this seems to require direct knowledge of the evolving density $p_t$, which is not available. The key simplification of flow matching is that one does not need to estimate the path density explicitly. Instead, one constructs a path through conditional interpolations between source and target samples and derives a corresponding conditional velocity field. Learning then becomes a regression problem toward that analytically known conditional target.
            """
        ),
        md(
            r"""
            This is a decisive conceptual step. The hard object is the full path density $p_t$, which would normally be expensive to characterize. Flow matching avoids that difficulty by shifting attention to conditional paths given paired endpoint samples. Once those conditional paths are chosen carefully, their instantaneous velocities are explicit. The model is then trained on those explicit local targets rather than on an intractable density-level objective.

            In this sense, flow matching continues a pattern we have already seen several times in the course. A generative problem that is difficult when formulated directly becomes manageable after introducing the right auxiliary structure. In VAEs that auxiliary structure was the variational distribution. In diffusion it was the noising process. In flow matching it is the conditional probability path.
            """
        ),
        md(
            r"""
            ## Conditional Probability Paths

            Let $\boldsymbol{x}_0 \sim p_0$ and $\boldsymbol{x}_1 \sim p_{gt}$. A simple example of conditional interpolation is the linear path $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$. Conditionally on the endpoints, this path has instantaneous velocity $\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{x}_1 - \boldsymbol{x}_0$. More generally, one may choose richer Gaussian or optimal-transport-inspired interpolations, but the central idea is unchanged: define a tractable conditional path and compute its conditional velocity exactly.

            A concrete picture is helpful. If the source sample lies near the origin and the target sample lies on one arm of a spiral-shaped dataset, the linear path simply draws a straight segment between them. Flow matching then learns how such many local straight-line suggestions should average into a coherent global transport field. This is why the method is richer than drawing one line between two points might first suggest.
            """
        ),
        md(
            r"""
            ```{admonition} Numerical Example: A Conditional Path in Two Dimensions
            :class: numerical-example

            Let the source point be $\boldsymbol{x}_0 = (0, 0)$ and the target point be $\boldsymbol{x}_1 = (4, 2)$. With the linear interpolation path $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$, the state at time $t = 0.25$ is $\boldsymbol{x}_{0.25} = (1, 0.5)$.

            The conditional velocity is constant: $\boldsymbol{u}_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \boldsymbol{x}_1 - \boldsymbol{x}_0 = (4, 2)$. So in this simple case the model is told that, at the point $(1, 0.5)$ and time $0.25$, the mass should keep moving in the direction $(4, 2)$. Training averages many such local instructions over many endpoint pairs until a global velocity field emerges.
            ```
            """
        ),
        md(
            r"""
            ```{prf:theorem} Conditional flow matching target
            :label: thm-cfm-target

            Let $(\boldsymbol{x}_0, \boldsymbol{x}_1)$ be sampled from a coupling between a source distribution $p_0$ and the target distribution $p_{gt}$. Let $\boldsymbol{x}_t$ be sampled from a conditional interpolation law $p_t(\boldsymbol{x} | \boldsymbol{x}_0, \boldsymbol{x}_1)$ with associated conditional velocity field $\boldsymbol{u}_t(\boldsymbol{x} | \boldsymbol{x}_0, \boldsymbol{x}_1)$. Then the minimizer of
            $$
            \mathcal{L}_{CFM}(\theta)
            =
            \mathbb{E}
            \left[
                \|
                    \boldsymbol{v}_\theta(\boldsymbol{x}_t, t)
                    -
                    \boldsymbol{u}_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)
                \|_2^2
            \right]
            $$
            is the conditional expectation
            $$
            \boldsymbol{v}_\theta^\star(\boldsymbol{x}, t)
            =
            \mathbb{E}
            \big[
                \boldsymbol{u}_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)
                \,\big|\,
                \boldsymbol{x}_t = \boldsymbol{x}
            \big].
            $$
            ```

            ```{prf:proof}
            This is a standard orthogonality property of squared-error regression. For fixed $(\boldsymbol{x}, t)$, the best predictor of the random target
            $\boldsymbol{u}_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1)$
            given the input $(\boldsymbol{x}_t, t)$ is its conditional expectation given those inputs. Therefore the minimizer of the global mean-squared error is exactly the conditional expectation field displayed above.
            ```
            """
        ),
        md(
            r"""
            The theorem is short, but it contains the practical power of flow matching. Instead of solving a hard density-matching problem directly, we turn learning into supervised regression against a conditional velocity that we can compute analytically from the chosen path. This is closely analogous to what happened in denoising diffusion: a difficult generative objective became a regression problem. The difference is that here the target is a velocity rather than a score or a noise vector.
            """
        ),
        md(
            r"""
            There is an important subtlety here. The learned field is not merely the raw conditional velocity for one specific pair of endpoints. The regression optimum is the conditional expectation of that target given the current intermediate state and time. This is why a single global vector field can emerge from many different sampled endpoint pairs. The network learns the average transport behavior compatible with the chosen path family.

            That observation also helps students see why flow matching is not trivial interpolation. The training data are generated from simple conditional constructions, but the resulting learned velocity field must generalize across the whole space and produce coherent global trajectories. The model is learning a transport law, not memorizing individual lines between points.
            """
        ),
        md(
            r"""
            ## Relation with Continuous Normalizing Flows

            Continuous normalizing flows also learn ODE dynamics that transform a source distribution into a target distribution. The crucial distinction is that CNFs are usually trained by maximum likelihood, which requires repeated evaluation of divergence terms and numerical integration inside the loss. Flow matching avoids this expensive likelihood-based training objective by prescribing a path and matching its velocity field directly. Sampling remains ODE-based, but training becomes much more straightforward.

            This is a major conceptual advantage for teaching. One can present flow matching as a deterministic generative model with neural ODE sampling, while keeping the learning rule as a simple regression problem. In that sense, it occupies an appealing middle ground: more directly transport-oriented than diffusion, but easier to train than classical likelihood-based CNFs.
            """
        ),
        md(
            r"""
            Put differently, CNFs and flow matching may use very similar samplers at generation time while differing substantially in how the vector field is trained. CNFs typically ask the model to be simultaneously a transport map and a tractable likelihood model. Flow matching drops the likelihood burden during training and focuses directly on matching transport targets. This often makes the optimization story easier to explain and, in many settings, easier to execute.
            """
        ),
        md(
            r"""
            ## Relation with Diffusion and Probability Flow ODEs

            The diffusion chapter already introduced the probability flow ODE associated with a stochastic forward process. Flow matching may be viewed as starting from the opposite end. Instead of deriving a deterministic transport equation from a stochastic model, it directly postulates a deterministic transport mechanism through a chosen probability path. This is why the two families are closely related. Both ultimately learn time-dependent vector fields that transport probability mass from a simple source to the data distribution. The main difference lies in how the vector field target is obtained.

            In score-based diffusion, the field is built from the score of perturbed densities. In flow matching, the field is built from a conditional path construction. This difference can have significant computational consequences. Flow matching can permit larger integration steps at sample time and avoids score-specific stochastic machinery during training, while still keeping a clean continuous-time viewpoint.
            """
        ),
        md(
            r"""
            This comparison is useful because it shows that diffusion and flow matching are not opponents so much as neighboring design choices in a shared space of transport-based generative models. Diffusion begins with a stochastic path and derives a usable vector field from score information. Flow matching begins with a deterministic path design and learns the corresponding field directly. The families therefore differ less in destination than in route.

            For a PhD audience, this is a good moment to emphasize that modern generative modeling is increasingly about path design, solver choice, and field parameterization rather than about one monolithic notion of "the generator." This shift in viewpoint is part of what makes the recent literature both rich and conceptually unifying.
            """
        ),
        md(
            r"""
            ## Gaussian and Optimal-Transport-Inspired Paths

            The linear interpolation path is conceptually simple, but it is not the only choice. One may define Gaussian bridges whose mean interpolates between source and target while the variance shrinks over time, or paths inspired by optimal transport couplings that encourage straighter, more efficient mass movement. The design of the path matters because it changes the regression target seen during training and therefore changes the geometry of the learned transport.

            From a teaching perspective, this is an excellent moment to emphasize that a generative model is never only a neural network architecture. It is also a choice of latent geometry, path design, and objective. In VAEs this geometry was hidden in the latent variable and the ELBO. In diffusion it was encoded in the noising process. In flow matching it is encoded directly in the probability path.
            """
        ),
        md(
            r"""
            The choice of path can have surprisingly concrete consequences. If the interpolation path is poorly aligned with the geometry of the data, the target velocity field may become unnecessarily curved or difficult to learn. If the path is closer to an optimal transport route, the learned trajectories may become straighter and numerically easier to integrate. This is why path design is not a decorative theoretical freedom. It is part of the effective inductive bias of the model.

            In practice, this also means that flow matching opens a larger design space than one first sees from the basic linear interpolation example. The training objective is simple, but the family of useful paths is rich. For students, this is a valuable reminder that simplicity of the loss does not imply narrowness of the modeling framework.
            """
        ),
        md(
            r"""
            ## Final Perspective for the Course

            Flow matching completes the conceptual arc of the course. We began with latent-variable models that emphasized tractable probabilistic surrogates. We then studied adversarial training, where a learned critic replaces explicit likelihoods. We next moved to diffusion, where generation becomes denoising along a stochastic path. Flow matching shows that one can keep the continuous-time transport viewpoint while removing the stochastic component and learning a deterministic velocity field directly through regression.

            The next notebook will implement a minimal flow matching example so that these ideas become concrete in code. The core modern reference is {cite}`lipman2023flow`, and the relation between transport design and more general couplings connects naturally to current work such as {cite}`tong2024improving`.
            """
        ),
        md(
            r"""
            Conceptually, this final chapter should leave the reader with a sharpened perspective on what the course has really been about. Across VAEs, GANs, diffusion, and flow matching, we repeatedly designed a simple source of randomness, a parameterized mechanism for transforming or interpreting it, and an objective that made learning feasible. The languages changed, but the structural question remained the same: how should probability mass be organized so that realistic images become learnable and samplable?

            Flow matching is therefore not just a recent add-on to the course. It is a particularly transparent endpoint of the larger narrative. Once students can see diffusion as path learning and flow matching as direct transport learning, the whole family of modern generative models becomes easier to compare at a conceptual level.
            """
        ),
    ],
)

write_notebook(
    "07-flow-matching/implementation.ipynb",
    [
        md(
            r"""
            # Flow Matching Implementation Notebook

            This notebook implements a minimal flow matching model on a toy two-dimensional dataset. The goal is pedagogical visibility. In two dimensions, the transport can be plotted directly, and students can see how trajectories move from a simple Gaussian source toward a structured target distribution. The code therefore complements the theory notebook more transparently than a full image-scale implementation would.

            The learning rule follows the core conditional flow matching pattern: sample source and target points, choose a time $t$, construct an interpolated point $\boldsymbol{x}_t$, compute the conditional velocity target analytically, and regress a neural network toward that target. Once training is complete, sampling becomes ODE integration from source samples through the learned velocity field.
            """
        ),
        md(
            r"""
            This notebook is intentionally two-dimensional because visibility matters more than realism here. In image-scale code, the same ideas are harder to see because trajectories are high-dimensional and the main evidence of success is the final sample. In two dimensions, we can inspect the point cloud and the transported trajectories directly. This makes the notebook especially valuable for teaching the mechanics of flow matching before students encounter larger models.
            """
        ),
        md(
            r"""
            ## Imports and a Toy Dataset

            To keep the notebook self-contained, we generate a synthetic mixture of circles in the plane. This type of dataset is common in flow-matching demos because it is simple enough to visualize but structured enough that transport is nontrivial.
            """
        ),
        md(
            r"""
            The dataset choice is deliberate. A unimodal Gaussian target would make the transport problem too easy and visually uninteresting. The two-ring structure forces the learned field to split mass meaningfully, which helps students see that the velocity network is learning geometry rather than merely drifting everything toward one central cluster.
            """
        ),
        code(
            """
            import math
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
            from tqdm.auto import tqdm
            import matplotlib.pyplot as plt

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(7)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(7)

            # The example is small enough to visualize directly after a short training run.
            batch_size = 512
            hidden_dim = 192
            time_dim = 96
            epochs = 4000
            lr = 8e-4


            def sample_target(batch_size, device):
                angles = 2 * math.pi * torch.rand(batch_size, device=device)
                radii = torch.where(
                    torch.rand(batch_size, device=device) > 0.5,
                    torch.full((batch_size,), 2.0, device=device),
                    torch.full((batch_size,), 4.0, device=device),
                )
                noise = 0.15 * torch.randn(batch_size, 2, device=device)
                # Two noisy rings give the transport field a visibly multimodal target.
                points = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)
                return points + noise


            def sample_source(batch_size, device):
                return torch.randn(batch_size, 2, device=device)
            """
        ),
        md(
            r"""
            ## Linear Conditional Path

            We use the simplest conditional interpolation:
            $$
            \boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1.
            $$
            Its conditional velocity is constant in time:
            $$
            \boldsymbol{u}_t(\boldsymbol{x}_t | \boldsymbol{x}_0, \boldsymbol{x}_1) = \boldsymbol{x}_1 - \boldsymbol{x}_0.
            $$
            This makes the implementation especially clean and keeps the focus on the flow-matching logic itself.
            """
        ),
        md(
            r"""
            This section is where the theory-to-code bridge is most direct. The path formula defines both the intermediate sample $\boldsymbol{x}_t$ and the target velocity $\boldsymbol{x}_1 - \boldsymbol{x}_0$. In other words, the entire supervised target comes from the chosen conditional interpolation. Students should notice how different this feels from GANs or VAEs: there is no discriminator, no ELBO, and no explicit density term in the code path that defines the learning signal.
            """
        ),
        code(
            """
            def sample_path(batch_size, device):
                x0 = sample_source(batch_size, device)
                x1 = sample_target(batch_size, device)
                t = torch.rand(batch_size, 1, device=device)
                # Linear conditional interpolation between source and target endpoints.
                xt = (1.0 - t) * x0 + t * x1
                # For the linear path, the conditional velocity is constant.
                ut = x1 - x0
                return x0, x1, xt, ut, t
            """
        ),
        md(
            r"""
            ## Time Embeddings and Velocity Network

            The network receives the current point and the current time, then predicts a two-dimensional velocity vector. Even here, sinusoidal time embeddings are useful because they let the model treat time as a rich continuous input rather than as a raw scalar.
            """
        ),
        md(
            r"""
            Conceptually, the role of time conditioning here is slightly different from diffusion even though the implementation pattern looks similar. In diffusion, time mostly indicates the noise level of a corrupted sample. In flow matching, time indicates where we are along a transport path from source to target. The network must therefore learn not only where mass should move, but when along the path particular movements should occur.
            """
        ),
        code(
            """
            class TimeEmbedding(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.dim = dim

                def forward(self, t):
                    half_dim = self.dim // 2
                    factor = math.log(10000.0) / max(half_dim - 1, 1)
                    frequencies = torch.exp(
                        torch.arange(half_dim, device=t.device) * -factor
                    )
                    angles = t * frequencies.unsqueeze(0)
                    emb = torch.cat([angles.sin(), angles.cos()], dim=1)
                    if self.dim % 2 == 1:
                        emb = F.pad(emb, (0, 1))
                    return emb


            class VelocityField(nn.Module):
                def __init__(self, hidden_dim=192, time_dim=96):
                    super().__init__()
                    self.time_embedding = nn.Sequential(
                        TimeEmbedding(time_dim),
                        nn.Linear(time_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    self.net = nn.Sequential(
                        nn.Linear(hidden_dim + 2, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, 2),
                    )

                def forward(self, x, t):
                    t_emb = self.time_embedding(t)
                    # Concatenate time context with the current spatial point.
                    return self.net(torch.cat([x, t_emb], dim=1))


            model = VelocityField(hidden_dim=hidden_dim, time_dim=time_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            """
        ),
        md(
            r"""
            ## Flow Matching Loss

            The training objective is simply mean squared error between the predicted velocity and the conditional target velocity. This is the implementation-level expression of the theorem proved in the theory notebook.
            """
        ),
        md(
            r"""
            This is the key practical payoff of the method. Once the path is chosen, training reduces to plain regression. That makes the optimization behavior easier to reason about than adversarial games and often simpler than likelihood-based neural ODE training. For students, it is valuable to recognize that the sophistication of the model lies more in the path design and the transport interpretation than in the loss expression itself.
            """
        ),
        code(
            """
            def flow_matching_loss(model, batch_size, device):
                _, _, xt, ut, t = sample_path(batch_size, device)
                pred = model(xt, t)
                return F.mse_loss(pred, ut)
            """
        ),
        md(
            r"""
            ## Training Loop

            Since the toy dataset is generated on the fly, there is no dataloader. Each iteration samples fresh source points, target points, and interpolation times. This keeps the code small and highlights the generative nature of the path construction.
            """
        ),
        md(
            r"""
            The on-the-fly sampling is pedagogically helpful because it makes clear that the target field is defined distributionally rather than from one fixed finite table of examples. Each optimization step sees a new set of endpoints and times, so the network learns an average transport law across the whole coupled sampling process. This reinforces the theoretical point that the learned field is a conditional expectation object, not a memorized per-sample map.
            """
        ),
        code(
            """
            history = []

            for step in tqdm(range(epochs), desc="Flow matching steps"):
                optimizer.zero_grad()
                loss = flow_matching_loss(model, batch_size, device)
                loss.backward()
                optimizer.step()

                history.append(loss.item())
                if (step + 1) % 200 == 0:
                    tqdm.write(f"Step {step + 1:04d} | loss: {loss.item():.6f}")
            """
        ),
        md(
            r"""
            ## ODE Sampling

            To sample from the trained model, we start from source points $\boldsymbol{x}_0 \sim p_0$ and integrate the learned ODE from time $0$ to time $1$. For transparency, we use the explicit Euler method. More accurate solvers can be substituted later without changing the learned velocity field.
            """
        ),
        md(
            r"""
            This section is a good opportunity to contrast learning difficulty with solver accuracy. The network may learn a good velocity field, yet a coarse Euler integrator can still produce visibly imperfect final samples if the step size is too large. This helps students see that continuous-time generative modeling has two numerical layers: fit the field well, then solve the resulting differential equation well enough to realize the intended transport.
            """
        ),
        code(
            """
            @torch.no_grad()
            def sample_trajectory(model, n_samples=2000, steps=100):
                model.eval()
                x = sample_source(n_samples, device)
                trajectory = [x.detach().cpu()]
                dt = 1.0 / steps

                for i in tqdm(range(steps), desc="ODE sampling", leave=False):
                    # Euler integration of the learned velocity field.
                    t = torch.full((n_samples, 1), i / steps, device=device)
                    v = model(x, t)
                    x = x + dt * v
                    trajectory.append(x.detach().cpu())

                return trajectory
            """
        ),
        code(
            """
            trajectory = sample_trajectory(model, n_samples=2000, steps=100)
            source_points = trajectory[0]
            final_points = trajectory[-1]
            target_points = sample_target(2000, device).cpu()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].scatter(source_points[:, 0], source_points[:, 1], s=4, alpha=0.5)
            axes[0].set_title("Source")
            axes[0].axis("equal")

            axes[1].scatter(final_points[:, 0], final_points[:, 1], s=4, alpha=0.5)
            axes[1].set_title("Generated")
            axes[1].axis("equal")

            axes[2].scatter(target_points[:, 0], target_points[:, 1], s=4, alpha=0.5)
            axes[2].set_title("Target")
            axes[2].axis("equal")

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            plt.figure(figsize=(7, 4))
            plt.plot(history)
            plt.xlabel("Training step")
            plt.ylabel("Flow matching loss")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            r"""
            ## How FID and KID Fit Flow Matching

            The current notebook is intentionally **not** an image-generation experiment. It learns a transport field in two dimensions, so FID and KID are not the natural quantitative tools here. Those metrics are designed for image distributions evaluated in a pretrained visual feature space.

            Still, for completeness, it is useful to show how the same evaluation pattern would look in an image-scale flow-matching model. The important point is that nothing about FID or KID is specific to VAEs, GANs, or diffusion. Once a model produces images, these metrics can compare the real and generated image distributions in exactly the same way.
            """
        ),
        code(
            """
            def prepare_for_inception_metrics(images):
                if images.size(1) == 1:
                    images = images.repeat(1, 3, 1, 1)
                return images.clamp(0.0, 1.0)


            @torch.no_grad()
            def compute_image_flow_fid_and_kid(flow_sampler, real_loader, device, num_fake=1000):
                fid = FrechetInceptionDistance(
                    feature=2048,
                    normalize=True,
                    reset_real_features=False,
                ).set_dtype(torch.float64).to(device)
                kid = KernelInceptionDistance(
                    feature=2048,
                    subsets=10,
                    subset_size=100,
                    normalize=True,
                    reset_real_features=False,
                ).to(device)

                for real_images, _ in real_loader:
                    real_images = prepare_for_inception_metrics(real_images.to(device))
                    fid.update(real_images, real=True)
                    kid.update(real_images, real=True)

                generated = 0
                while generated < num_fake:
                    batch_n = min(128, num_fake - generated)
                    fake_images = flow_sampler(batch_n).to(device)
                    fake_images = prepare_for_inception_metrics(fake_images)
                    fid.update(fake_images, real=False)
                    kid.update(fake_images, real=False)
                    generated += batch_n

                kid_mean, kid_std = kid.compute()
                return {
                    "fid": fid.compute().item(),
                    "kid_mean": kid_mean.item(),
                    "kid_std": kid_std.item(),
                }
            """
        ),
        md(
            r"""
            In the present two-dimensional toy setting, better diagnostics are geometric ones: do the transported samples reach both rings, do trajectories look smooth, and does the generated cloud align with the target cloud. But if one later swaps this toy example for an image-scale flow-matching notebook, the metric block above can be reused almost unchanged.
            """
        ),
        md(
            r"""
            ## Practical Discussion

            Even though this example is only two-dimensional, all the main elements of flow matching are present: a source distribution, a target distribution, a conditional path, an analytically known velocity target, a time-conditioned neural vector field, and ODE-based sampling. The image-scale version changes the architecture and the dataset, but not the conceptual structure.

            This notebook is also a useful comparison point with diffusion. In diffusion, we learned to denoise along a stochastic path. Here we learn to move deterministically along a prescribed transport path. The objectives are both regression problems, but the targets are different: diffusion predicts noise or scores, whereas flow matching predicts velocities.
            """
        ),
        md(
            r"""
            When this notebook works well, the sampled trajectories should fan out from the source Gaussian toward the two-ring target in a way that looks smooth rather than erratic. If the final samples miss one ring, the model may be undertrained or the hidden dimension too small. If trajectories look jagged or unstable, the Euler step count may be too low even if the learned field is reasonable. If the loss decreases but the geometry remains poor, the issue may be the simplicity of the chosen linear path rather than the optimizer itself.

            The larger conceptual lesson is that flow matching makes transport visible. Students can literally watch probability mass move. That makes this notebook a strong capstone implementation for the course because it reveals, in a minimal setting, the transport viewpoint that had been building implicitly through diffusion and probability flow ODEs.
            """
        ),
    ],
)

write_notebook(
    "08-references/further-reading.ipynb",
    [
        md(
            r"""
            # Further Reading

            These notes cite the core sources used throughout the course, but students often benefit from a small amount of reading guidance rather than a flat bibliography. A good way to use the references is to distinguish between background texts, canonical first papers for each model family, and more advanced follow-up papers that reveal how the field evolved.

            For broad background, {cite}`bishop2006pattern` remains a strong source for probabilistic modeling language, while {cite}`goodfellow2016deep` is still a useful general deep-learning reference for neural architectures, optimization, and representation learning. Students who feel less comfortable with the foundations chapters should begin there before diving more deeply into the specialized generative-model papers.
            """
        ),
        md(
            r"""
            ## Suggested Reading Order

            A sensible reading path after the course is the following.

            First, for latent-variable generative modeling, read the original VAE papers {cite}`kingma2013auto` and {cite}`rezende2014stochastic`. They are concise and reveal the core variational logic that later reappears in more sophisticated models. Next, for adversarial learning, read {cite}`goodfellow2014generative`, then compare it with stabilization-oriented work such as {cite}`arjovsky2017wasserstein`, {cite}`miyato2018spectral`, and {cite}`zhu2017unpaired` to see how quickly the field moved from a simple game formulation to a broader family of objectives and constraints.

            For diffusion, a good route is to read {cite}`ho2020denoising` first for the discrete DDPM perspective, then {cite}`song2021scorebased` for the continuous-time score-based view, and then {cite}`song2020ddim` for deterministic sampling. Students interested in modern large-scale systems should follow these with {cite}`rombach2022high`, which shows how latent-space diffusion made high-resolution synthesis far more practical.
            """
        ),
        md(
            r"""
            ## Reading by Chapter Theme

            The chapters of the book line up naturally with particular papers. The probabilistic latent-variable thread is anchored by {cite}`kingma2013auto` and {cite}`rezende2014stochastic`. The adversarial thread begins with {cite}`goodfellow2014generative` and branches into discrepancy design and stabilization via {cite}`arjovsky2017wasserstein` and {cite}`miyato2018spectral`. The stochastic transport and score-based thread is represented by {cite}`ho2020denoising`, {cite}`song2020denoising`, {cite}`song2021scorebased`, and {cite}`song2021maximum`. The deterministic transport and flow-matching thread is represented by {cite}`lipman2023flow` and more recent path-design work such as {cite}`tong2024improving`.

            A useful meta-question when reading any of these papers is the same one that structures the course itself: what is the chosen source of randomness, what object is the neural network asked to predict or represent, and what surrogate objective makes learning feasible? Asking those three questions repeatedly makes the literature much easier to organize.
            """
        ),
        md(
            r"""
            ## How To Read the Papers

            Students reading these references for the first time should not expect every proof or implementation detail to be equally central. A productive first pass is usually:

            1. identify the probabilistic objects being modeled;
            2. identify the trainable neural components;
            3. identify the obstacle to direct maximum-likelihood learning;
            4. identify the surrogate objective or transport reformulation that resolves that obstacle.

            Once that scaffold is clear, the paper's specific derivations and engineering choices become much easier to interpret. This is especially important for diffusion and flow-matching papers, where the notation can look heavy before the conceptual structure is visible.

            ```{bibliography}
            ```
            """
        )
    ],
)
