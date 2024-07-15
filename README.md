# Hand-drawn Map Navigation (HAM-Nav) Framework

## Overview
The Hand-drawn Map Navigation (HAM-Nav) framework uses multi-modal large language models (LLMs) to navigate robots with hand-drawn maps. These maps are intuitive and useful in dynamic or hazardous environments, addressing the inaccuracies and abstract nature of human sketches for precise robot navigation.

## Motivation
Traditional navigation methods require accurate maps, which are costly and time-consuming to obtain. In scenarios like search and rescue or construction inspections, maintaining accurate maps is impractical due to frequent changes. Hand-drawn maps provide a cost-effective alternative by offering prior spatial information without the need for expensive map acquisition.

Hand-drawn maps offer a middle ground between map-based navigation, which relies on precise and often difficult-to-maintain maps, and map-less navigation, which can be inefficient and error-prone due to the lack of prior environmental knowledge.

## Challenges
- __Inherent Inaccuracies__: Hand-drawn maps can be imprecise, with distortions and scale issues that differ from the actual environment.
- __Abstract Nature__: Human sketches often omit details or simplify complex environments, making it difficult for robots to interpret them accurately. Furthermore, users may unintentionally omit important environmental features from the map, such as critical obstacles or key navigational landmarks, considering them irrelevant for navigation.
- __Lack of Standardization__: Hand-drawn maps vary widely in style and detail, requiring sophisticated interpretation methods to ensure reliable navigation. These maps exhibit high intra-class variations due to individual drawing styles and large inter-domain variations between the sketches and the actual environment.

## Ongoing Work and Code Release
Please note that the work on the HAM-Nav framework is ongoing. This repository currently contains only a snippet of the codebase. The full code will be released once the associated paper is accepted for publication. We appreciate your understanding and patience. For any inquiry, please email: aaronhao.tan@utoronto.ca