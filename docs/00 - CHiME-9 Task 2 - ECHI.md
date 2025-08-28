https://www.chimechallenge.org/current/task2/index
## Enhancing Conversations to address Hearing Impairment.

**Hearing impairment is a rapidly escalating global challenge**. The World Health Organization (WHO) projects that by 2050, [2.5 billion people](https://www.who.int/publications/i/item/9789240020481#:~:text=The%20World%20Report%20on%20Hearing,into%20their%20national%20health%20plans.) will experience some form of hearing loss, with 700 million requiring rehabilitation. Even mild hearing loss can make navigating conversations in noisy environments, such as bustling bars or crowded restaurants, incredibly difficult and exhausting. This is a reality many adults face from around the age of 40 onwards.

Despite advancements, modern hearing aids still struggle to consistently isolate and enhance target speech in these dynamic, real-world acoustic settings. However, the emergence of new low-power Deep Neural Network (DNN) chips and other technological breakthroughs is paving the way for more sophisticated signal processing algorithms. These innovations hold immense potential to transform assistive listening technologies.

### CHiME-9 Task 2: The ECHI Challenge

This task invites participants to tackle the challenge of **Enhancing Conversations to address Hearing Impairment (ECHI)**. We consider signal processing that enables a device such as illustrated below, which will block out unwanted noise but while still allowing the wearer to hear their conversational partners.

![Illustrating the ECHI concept](https://www.chimechallenge.org/current/task2/images/echi_concept.png)

Image credit: Tom Kit Barker / (c) 2025

To enable the development of algorithms that solve this task, we provide a unique dataset of real conversations captured in a carefully simulated noisy cafeteria environment. These recordings feature a complex mix of surrounding distractor conversations and diffuse cafeteria noise.

The conversations were recorded using both multichannel **hearing aid devices** and **smart glasses** worn by the participants. Entrants are asked to develop solutions for both devices. Each device should be processed separately using **real-time, low-latency** algorithms. Your goal is to produce a single-channel audio stream for each conversational partner. These enhanced streams should be effectively free of all background noise and competing speech, while preserving the target speaker’s voice with minimal distortion. The ultimate aim is to enable these processed streams to be re-summed, yielding an enhanced rendition of the conversation where all distracting room noise is seamlessly “faded out.”

### Evaluation

Challenge entrants will be evaluated using a held-out test set, featuring conversations and talkers entirely unseen during the training or development phases. Your enhanced signals will be assessed through a dual approach:

- **Objective Metrics**: A comprehensive suite of objective metrics will evaluate the quality and intelligibility of each speech segment within each stream.
- **Subjective Listening Tests**: Crucially, human listening tests will be conducted to directly assess the perceived quality and intelligibility of the entire conversation when the individual enhanced streams are re-summed.

To support your development, we will provide a baseline system and a dedicated development dataset. These resources, combined with the objective metrics, will allow you to gauge your system’s performance throughout the development cycle.

### How to Get Started

Ready to contribute to this vital challenge? All the resources you need to begin your journey here on this site:

- [Data Overview](https://www.chimechallenge.org/current/task2/data): Find comprehensive details about the dataset and instructions for downloading it.
- [Rules and Evaluation](https://www.chimechallenge.org/current/task2/rules): Understand the full rules of the challenge, including system constraints and how your submissions will be evaluated.
- [Software & Baseline](https://www.chimechallenge.org/current/task2/baseline): Explore the provided software, including our baseline system and the evaluation framework, with a direct link to the GitHub repository.
- [Submission Guide](https://www.chimechallenge.org/current/task2/submission): Learn how to prepare and submit your enhanced audio streams for evaluation.
- [Frequently Asked Questions (FAQ)](https://www.chimechallenge.org/current/task2/faq): A growing resource compiling answers to common questions from the community.