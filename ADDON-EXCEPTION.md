# Coefflow Add-on Exception

Copyright (C) 2026 Gregor Burroughs

Coefflow is licensed under the GNU Affero General Public License, version 3 (the "AGPL"), with the following additional permission under section 7 of the AGPL.

## Definitions

"Coefflow" means the software in this repository and any version of it licensed under the AGPL with this exception.

An "Add-on" is a work that:

1. is an add-on pack (a folder holding a `pack.py` and one or more module folders, each with a `module.py`) or a single module (a `module.py` and any files it relies on), written to be loaded by Coefflow through its add-on or module loading mechanism;
2. interacts with Coefflow only by being loaded through that mechanism and by importing and calling Coefflow's published code, such as `engine` and `ui`; and
3. contains no part of Coefflow, copied or modified, beyond the names, settings and function signatures needed to follow the add-on and module interface.

## Permission

As a special exception, the copyright holder gives you permission to write, use, modify and convey Add-ons, and to run them together with Coefflow (including letting users interact with that combination over a network), under terms of your choice, including proprietary terms. An Add-on does not become subject to the AGPL because it is loaded into the same process as Coefflow or forms a combined work with it.

## Conditions

1. Coefflow itself stays under the AGPL. If you modify Coefflow, convey it, or let users interact with a modified version of it over a network, the AGPL applies to Coefflow and your modifications as normal. This exception covers only Add-ons.
2. A change to Coefflow made so that an Add-on works, such as a new hook in `engine/`, is a modification of Coefflow and not part of the Add-on.

## Modified versions

If you modify Coefflow, you may extend this exception to your version, but you are not obliged to. If you do not wish to, delete this exception from your version.
