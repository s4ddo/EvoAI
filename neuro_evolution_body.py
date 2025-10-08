
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from typing import TYPE_CHECKING, Any, Literal
if TYPE_CHECKING:
    from networkx import DiGraph
from pathlib import Path
import numpy as np


RNG = np.random.default_rng(1)
NUM_OF_MODULES = 30

# --- DATA SETUP ---
SCRIPT_NAME = "test"
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)



def make_random_robot(fixed_genotype = []):
     # ? ------------------------------------------------------------------ #

    genotype = fixed_genotype
    if not fixed_genotype:
        genotype_size = 64
        type_p_genes = RNG.random(genotype_size).astype(np.float32)
        conn_p_genes = RNG.random(genotype_size).astype(np.float32)
        rot_p_genes = RNG.random(genotype_size).astype(np.float32)

        genotype = [
            type_p_genes,
            conn_p_genes,
            rot_p_genes,
        ]

    
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    save_graph_as_json(
        robot_graph,
        DATA / "robot_graph.json",
    )

    # ? ------------------------------------------------------------------ #
    # Print all nodes
    return construct_mjspec_from_graph(robot_graph), genotype

