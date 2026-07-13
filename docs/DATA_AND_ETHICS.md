# Data and Ethics

## Source

The pipeline accesses the public CFPB consumer complaint collection through the Hugging Face repository `Mouwiya/cfpb-consumer-complaints`.

## Responsible use

Although the narratives are publicly released and generally redacted by the source provider, they describe real financial experiences. Users should:

- avoid attempts to identify complainants;
- avoid reconstructing masked information;
- avoid using narratives for decisions about individuals;
- preserve source attribution;
- review current source terms before redistribution or commercial use;
- treat representative complaints as research evidence rather than entertainment content.

## Repository policy

Raw, cleaned, representative-document, and fully clustered narrative files are generated locally and excluded from this public repository. Aggregate tables and figures are committed for reproducibility while reducing unnecessary redistribution of complaint text.

## Analytical limitations

The first 50,000 streamed records are not a random sample of the complete CFPB collection. The analysis is exploratory and should not be interpreted as a population estimate or as an assessment of any company, product, or consumer group.
