#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fire

import speakerbox_analysis

###############################################################################


def main() -> None:
    fire.Fire(speakerbox_analysis, name="speakerbox-analysis")


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
