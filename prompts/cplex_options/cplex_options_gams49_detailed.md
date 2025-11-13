advind (integer): advanced basis use ↵

Use an Advanced Basis. GAMS/Cplex will automatically use an advanced basis from a previous solve statement. The GAMS BRatio option can be used to specify when not to use an advanced basis. The Cplex option AdvInd can be used to ignore a basis passed on by GAMS (it overrides BRatio).

Default: determined by GAMS Bratio

value	meaning
0	Do not use advanced basis
1	Use advanced basis if available
2	Crash an advanced basis if available (use basis with presolve)
aggcutlim (integer): aggregation limit for cut generation ↵

Limits the number of constraints that can be aggregated for generating flow cover and mixed integer rounding cuts. For most purposes, the default will be satisfactory.

Default: 3

aggfill (integer): aggregator fill parameter ↵

Aggregator fill limit. If the net result of a single substitution is more non-zeros than the setting of the AggFill parameter, the substitution will not be made.

Default: 10

aggind (integer): aggregator on/off ↵

This option, when set to a nonzero value, will cause the Cplex aggregator to use substitution where possible to reduce the number of rows and columns in the problem. If set to a positive value, the aggregator will be applied the specified number of times, or until no more reductions are possible. At the default value of -1, the aggregator is applied once for linear programs and an unlimited number of times for mixed integer problems.

Default: -1

value	meaning
-1	Once for LP, unlimited for MIP
0	Do not use
>0	Aggregator will be applied the specified number of times
auxrootthreads (integer): number of threads for auxiliary tasks at the root node ↵

Partitions the number of threads for CPLEX to use for auxiliary tasks while it solves the root node of a problem. On a system that offers N processors or N global threads, if you set this parameter to n, where N>n>0 then CPLEX uses at most n threads for auxiliary tasks and at most N-n threads to solve the root node. See also the parameter Threads.

You cannot set n, the value of this parameter, to a value greater than or equal to N, the number of processors or global threads offered on your system. In other words, when you set this parameter to a value other than its default, that value must be strictly less than the number of processors or global threads on your system. Independent of the auxiliary root threads parameter, CPLEX will never use more threads than those defined by the global default thread count parameter. CPLEX also makes sure that there is at least one thread available for the main root tasks. For example, if you set the global threads parameter to 3 and the auxiliary root threads parameter to 4, CPLEX still uses only two threads for auxiliary root tasks in order to keep one thread available for the main root tasks. At its default value, 0 (zero), CPLEX automatically chooses the number of threads to use for the primary root tasks and for auxiliary tasks. The number of threads that CPLEX uses to solve the root node depends on several factors: 1) the number of processors available on your system; 2) the number of threads available to your application on your system (for example, as a result of limited resources or competition with other applications); 3) the value of the global default thread count parameter Threads.

Default: 0

value	meaning
-1	Off: do not use additional threads for auxiliary tasks
0	Automatic: let CPLEX choose the number of threads to use
N>n>0	Use n threads for auxiliary root tasks
baralg (integer): algorithm selection ↵

Selects which barrier algorithm to use. The default setting of 0 uses the infeasibility-estimate start algorithm for LPs and MIP subproblems and the standard barrier algorithm, option 3, for other cases. The standard barrier algorithm is almost always fastest. The alternative algorithms, options 1 and 2, may eliminate numerical difficulties related to infeasibility, but will generally be slower.

Default: 0

value	meaning
0	Same as 1 for LPs and MIP subproblems, 3 otherwise
1	Infeasibility-estimate start
2	Infeasibility-constant start
3	standard barrier algorithm
barcolnz (integer): dense column handling ↵

Determines whether or not columns are considered dense for special barrier algorithm handling. At the default setting of 0, this parameter is determined dynamically. Values above 0 specify the number of entries in columns to be considered as dense.

Default: 0

barcrossalg (integer): barrier crossover method ↵

Selects which crossover method is used at the end of a barrier optimization. To turn off crossover set SolutionType to 2.

Default: 0

value	meaning
0	Automatic
1	Primal crossover
2	Dual crossover
bardisplay (integer): progress display level ↵

Determines the level of progress information to be displayed while the barrier method is running.

Default: 1

value	meaning
0	No progress information
1	Display normal information
2	Display diagnostic information
barepcomp (real): convergence tolerance ↵

Determines the tolerance on complementarity for convergence of the barrier algorithm. The algorithm will terminate with an optimal solution if the relative complementarity is smaller than this value.

Default: 1.0e-08

bargrowth (real): unbounded face detection ↵

Used by the barrier algorithm to detect unbounded optimal faces. At higher values, the barrier algorithm will be less likely to conclude that the problem has an unbounded optimal face, but more likely to have numerical difficulties if the problem does have an unbounded face.

Default: 1.0e+12

baritlim (integer): iteration limit ↵

Determines the maximum number of iterations for the barrier algorithm. When set to 0, no Barrier iterations occur, but problem setup occurs and information about the setup is displayed (such as Cholesky factorization information). When left at the default value, there is no explicit limit on the number of iterations.

Default: large

barmaxcor (integer): maximum correction limit ↵

Specifies the maximum number of centering corrections that should be done on each iteration. Larger values may improve the numerical performance of the barrier algorithm at the expense of computation time. The default of -1 means the number is automatically determined.

Default: -1

barobjrng (real): maximum objective function ↵

Determines the maximum absolute value of the objective function. The barrier algorithm looks at this limit to detect unbounded problems.

Default: 1.0e+20

barorder (integer): row ordering algorithm selection ↵

Determines the ordering algorithm to be used by the barrier method. By default, Cplex attempts to choose the most effective of the available alternatives. Higher numbers tend to favor better orderings at the expense of longer ordering run times. The automatic option includes additional processing and may yield results that differ from the explicit choice.

Default: 0

value	meaning
0	Automatic
1	Approximate Minimum Degree (AMD)
2	Approximate Minimum Fill (AMF)
3	Nested Dissection (ND)
barqcpepcomp (real): convergence tolerance for the barrier optimizer for QCPs ↵

Range: [1.0e-12, 1.0e+75]

Default: 1.0e-07

barstartalg (integer): barrier starting point algorithm ↵

This option sets the algorithm to be used to compute the initial starting point for the barrier solver. The default starting point is satisfactory for most problems. Since the default starting point is tuned for primal problems, using the other starting points may be worthwhile in conjunction with the PreDual parameter.

Default: 1

value	meaning
1	default primal, dual is 0
2	default primal, estimate dual
3	primal average, dual is 0
4	primal average, estimate dual
bbinterval (integer): best bound interval ↵

Set interval for selecting a best bound node when doing a best estimate search. Active only when NodeSel is 2 (best estimate). Decreasing this interval may be useful when best estimate is finding good solutions but making little progress in moving the bound. Increasing this interval may help when the best estimate node selection is not finding any good integer solutions. Setting the interval to 1 is equivalent to setting NodeSel to 1.

Default: 7

bendersfeascuttol (real): Tolerance for whether a feasibility cut has been violated in Benders decomposition ↵

Default: 1.0e-06

bendersoptcuttol (real): Tolerance for optimality cuts in Benders decomposition ↵

Default: 1.0e-06

.benderspartition (integer): Benders partition ↵

Default: 0

benderspartitioninstage (boolean): Benders partition through stage variable suffix ↵

Default: 0

bendersstrategy (integer): Benders decomposition algorithm as a strategy ↵

Given a formulation of a problem, CPLEX can decompose the model into a single master and (possibly multiple) subproblems. To do so, CPLEX can make use of annotations that you supply for your model. The strategy can be applied to mixed-integer linear programs (MILP). For certain types of problems, this approach offers significant performance improvements as subproblems can be solved in parallel.

For mixed integer programs (MIP), under certain conditions, CPLEX can apply Benders algorithm to improve the search to find more feasible solutions more quickly.

Default: 0

value	meaning
-1	Off
Execute conventional branch and bound; ignore any Benders annotations. That is, do not use Benders algorithm even if a Benders partition of the current model is present
0	Automatic
If annotations specifying a Benders partition of the current model are available, CPLEX attempts to decompose the model. CPLEX uses the master as given by the annotations, and attempts to partition the subproblems further, if possible, before applying Benders algorithm to solve the model. If the user supplied annotations, but the annotations supplied do not lead to a complete decomposition into master and disjoint subproblems (that is, if the annotations are wrong in that sense), CPLEX produces an error.
1	Apply user annotations
CPLEX applies Benders algorithm to a decomposition based on annotations supplied by the user. If no annotations to decompose the model are available, this setting produces an error. If the user supplies annotations, but the supplied annotations do not lead to a complete partition of the original model into disjoint master and subproblems, then this setting produces an error.
2	Apply user annotations with automatic support for subproblems
CPLEX accepts the master as given and attempts to decompose the remaining elements into disjoint subproblems to assign to workers. It then solves the Benders decomposition of the model. If no annotations to decompose the model are available, this setting produces an error. If the user supplies annotations, but the supplied annotations do not lead to a complete partition of the original model into disjoint master and subproblems, then this setting produces an error.
3	Apply automatic decomposition
CPLEX ignores any annotation supplied with the model; CPLEX applies presolve; CPLEX then automatically generates a Benders partition, putting integer variables in master and continuous linear variables into disjoint subproblems. CPLEX then solves the Benders decomposition of the model. If the problem is a strictly linear program (LP), that is, there are no integer-constrained variables to put into master, then CPLEX reports an error. If the problem is a mixed integer linear program (MILP) where all variables are integer-constrained, (that is, there are no continuous linear variables to decompose into disjoint subproblems) then CPLEX reports an error.
bndrng (string): do lower / upper bound ranging ↵

Calculate sensitivity ranges for the specified GAMS lower and upper bounds. Unlike most options, BNDRng can be repeated multiple times in the options file. Sensitivity range information will be produced for each GAMS lower and upper bound named. Specifying all will cause range information to be produced for all lower and upper bounds. Range information will be printed to the beginning of the solution listing in the GAMS listing file unless option RngRestart is specified.

bndstrenind (integer): bound strengthening ↵

Use bound strengthening when solving mixed integer problems. Bound strengthening tightens the bounds on variables, perhaps to the point where the variable can be fixed and thus removed from consideration during the branch and bound algorithm. This reduction is usually beneficial, but occasionally, due to its iterative nature, takes a long time.

Default: -1

value	meaning
-1	Determine automatically
0	Don't use bound strengthening
1	Use bound strengthening
bqpcuts (integer): boolean quadric polytope cuts for nonconvex QP or MIQP solved to global optimality ↵

Default: 0

value	meaning
-1	Do not generate BQP cuts
0	Determined automatically
1	Generate BQP cuts moderately
2	Generate BQP cuts aggressively
3	Generate BQP cuts very aggressively
brdir (integer): set branching direction ↵

Used to decide which branch (up or down) should be taken first at each node.

Default: 0

value	meaning
-1	Down branch selected first
0	Algorithm decides
1	Up branch selected first
bttol (real): backtracking limit ↵

This option controls how often backtracking is done during the branching process. At each node, Cplex compares the objective function value or estimated integer objective value to these values at parent nodes; the value of the bttol parameter dictates how much relative degradation is tolerated before backtracking. Lower values tend to increase the amount of backtracking, making the search more of a pure best-bound search. Higher values tend to decrease the amount of backtracking, making the search more of a depth-first search. This parameter is used only once a first integer solution is found or when a cutoff has been specified.

Range: [0.0, 1.0]

Default: 1.0

calcqcpduals (integer): calculate the dual values of a quadratically constrained problem ↵

Default: 1

value	meaning
0	Do not calculate dual values
1	Calculate dual values as long as it does not interfere with presolve reductions
2	Calculate dual values and disable any presolve reductions that would interfere
cardls (integer): decides how often to apply the cardinality local search heuristic (CLSH) ↵

Default: -1

value	meaning
-1	Do not apply CLSH
0	Automatic
1	Apply the CLSH only at the root node
2	Apply the CLSH at the nodes of the branch and bound tree
cliques (integer): clique cut generation ↵

Determines whether or not clique cuts should be generated during optimization.

Default: 0

value	meaning
-1	Do not generate clique cuts
0	Determined automatically
1	Generate clique cuts moderately
2	Generate clique cuts aggressively
3	Generate clique cuts very aggressively
clocktype (integer): clock type for computation time ↵

Decides how computation times are measured for both reporting performance and terminating optimization when a time limit has been set. Small variations in measured time on identical runs may be expected on any computer system with any setting of this parameter.

Default: 2

value	meaning
0	Automatic
1	CPU time
2	Wall clock time
clonelog (integer): enable clone logs ↵

The clone logs contain information normally recorded in the ordinary log file but inconvenient to send through the normal log channel in case of parallel execution. The information likely to be of most interest to you are special messages, such as error messages, that result from calls to the LP optimizers called for the subproblems. The clone log files are named cloneK.log, where K is the index of the clone, ranging from 0 (zero) to the number of threads minus one. Since the clones are created at each call to a parallel optimizer and discarded when it exits, the clone logs are opened at each call and closed at each exit. The clone log files are not removed when the clones themselves are discarded.

Default: 0

value	meaning
-1	Clone log files off
0	Automatic
1	Clone log files on
coeredind (integer): coefficient reduction on/off ↵

Coefficient reduction is a technique used when presolving mixed integer programs. The benefit is to improve the objective value of the initial (and subsequent) linear programming relaxations by reducing the number of non-integral vertexes. However, the linear programs generated at each node may become more difficult to solve.

Default: -1

value	meaning
-1	Automatic
0	Do not use coefficient reduction
1	Reduce only to integral coefficients
2	Reduce all potential coefficients
3	Reduce aggressively with tilting
conflictalg (integer): algorithm CPLEX uses in the conflict refiner to discover a minimal set of conflicting constraints in an infeasible model ↵

Default: 0

conflictdisplay (integer): decides how much information CPLEX reports when the conflict refiner is working ↵

Default: 1

value	meaning
0	No output
1	Summary display
2	Detailed display
covers (integer): cover cut generation ↵

Determines whether or not cover cuts should be generated during optimization.

Default: 0

value	meaning
-1	Do not generate cover cuts
0	Determined automatically
1	Generate cover cuts moderately
2	Generate cover cuts aggressively
3	Generate cover cuts very aggressively
cpumask (string): switch and mask to bind threads to processors (Linux only) ↵

The value of this parameter serves as a switch to turn on (or to turn off) CPLEX binding of multiple threads to multiple processors on platforms where this feature is available. Hexadecimal values of this parameter serve as a mask to specify to CPLEX which processors (or cores) to use in binding multiple threads to multiple processors. CPU binding is also sometimes known as processor affinity. CPU binding reduces the variability of CPLEX runs. On some occasions, running the same CPLEX on the same (non trivial) models would produce a big variation in runtime, e.g. 1000 seconds versus 900 seconds on a 12 core machine. These differences happen while CPLEX still gets exactly the same results and executes the exact same path, thanks to its completely deterministic algorithms. Running the same tests with CPU binding enabled reduced this variability in running time significantly.

If not set to off or auto CPLEX treats the value of this parameter as a string that resembles a hexadecimal number without the usual 0x prefix. A valid string consists of these elements: a) any digit from 0 (zero) through 9 (inclusive), b) any lower case character in the range a through f (inclusive), and c) any upper case character in the range A through F (inclusive). CPLEX rejects a string containing any other digits or characters than those.

When the value of this parameter is a valid string, each bit of this string corresponds to a central processing unit (CPU), that is, to a processor or core. The lowest order bit of the string corresponds to the first logical CPU, and the highest order corresponds to the last logical CPU. For example, 00000001 designates processor #0, 00000003 designates processors #0 and #1, FFFFFFFF designates all processors #0 through #31. CPLEX uses the ith CPU if and only if the ith bit of this string is set to 1 (one). Tip: For GNU/Linux users, this parameter behaves like the taskset command (except that this parameter lacks the prefix 0x).

If this CPU mask parameter is set to a valid string that designates a hexadecimal number, but global Threads count is set to 0 (zero), then CPLEX still starts as many threads as the number of cores on the machine, but only the cores enabled in the mask will be used.

For example, if a user sets this CPU mask parameter to the hexadecimal value "f" on a 16-core machine, and the user sets the global Threads count to 0 (zero), the result is 16 threads. These 16 threads will be bound to the first four cores in a round-robin way: treads 1,5,9,13 to core 1, threads 2,6,10,14 to core 2 and so on. This situation is probably not what the user intended. Therefore, if you set this CPU mask parameter, then you should also set global threads count; indeed, you should set the threads parameter to the number of active cores designated by the mask.

For example, on a 16 core machine, consider the difference between the value "off" and the value ffff. If the value of this parameter is "off" CPLEX does no binding. If the value of this parameter is ffff, CPLEX binds threads to cores.

Default: auto

value	meaning
auto	CPLEX decides whether to bind threads to cores (or processors)
off	CPLEX performs no binding
hex	CPLEX binds the threads in round-robin fashion to the cores specified by the mask
craind (integer): crash strategy (used to obtain starting basis) ↵

The crash option biases the way Cplex orders variables relative to the objective function when selecting an initial basis.

Default: 1

value	meaning
-1	Primal: alternate ways of using objective coefficients. Dual: aggressive starting basis
0	Primal: ignore objective coefficients during crash. Dual: aggressive starting basis
1	Primal: alternate ways of using objective coefficients. Dual: default starting basis
cutlo (real): lower cutoff for tree search ↵

Sets the lower cutoff tolerance. When the problem is a maximization problem, CPLEX cuts off or discards solutions that are less than the specified cutoff value. If the model has no solution with an objective value greater than or equal to the cutoff value, then CPLEX declares the model infeasible. In other words, setting the lower cutoff value c for a maximization problem is similar to adding this constraint to the objective function of the model: obj>=c.

This option overrides the GAMS Cutoff setting.

This parameter is not effective with FeasOpt. FeasOpt cannot analyze an infeasibility introduced by this parameter. If you want to analyze such a condition, add an explicit objective constraint to your model instead.

Default: -1.0e+75

cutpass (integer): maximum number of cutting plane passes ↵

Sets the upper limit on the number of passes that will be performed when generating cutting planes on a mixed integer model.

Default: 0

value	meaning
-1	None
0	Automatically determined
>0	Maximum passes to perform
cuts (string): default cut generation ↵

Allows generation setting of all optional cuts at once. This is done by changing the meaning of the default value (0: automatic) for the various Cplex cut generation options. The options affected are Cliques, Covers, DisjCuts, FlowCovers, FlowPaths, FracCuts, GUBCovers, ImplBd, LiftProjCuts, MCFCuts, MIRCuts, and Symmetry.

Default: 0

value	meaning
-1	Do not generate cuts
0	Determined automatically
1	Generate cuts moderately
2	Generate cuts aggressively
3	Generate cuts very aggressively
4	Generate cuts highly aggressively
5	Generate cuts extremely aggressively
cutsfactor (real): cut limit ↵

This option limits the number of cuts that can be added. For values between zero and one inclusive (that is, in the range [0.0, 1.0], CPLEX generates no cuts.

For values strictly greater than 1.0 (one), CPLEX limits the number of rows in the model with cuts added.

The limit on this total is the product of CutsFactor times the original number of rows. If CPLEX has presolved the model, the original number of rows is the number of rows in the presolved model. (This behavior with respect to a presolved model is unchanged.)

CPLEX regards negative values of this parameter as equivalent to the default value -1.0. That is, a negative value specifies no particular limit on the number of cuts. CPLEX computes and dynamically adjusts such a limit automatically

Default: -1.0

cutup (real): upper cutoff for tree search ↵

Sets the upper cutoff tolerance. When the problem is a minimization problem, CPLEX cuts off or discards any solutions that are greater than the specified upper cutoff value. If the model has no solution with an objective value less than or equal to the cutoff value, CPLEX declares the model infeasible. In other words, setting an upper cutoff value c for a minimization problem is similar to adding this constraint to the objective function of the model: obj<=c.

This option overrides the GAMS Cutoff setting.

This parameter is not effective with FeasOpt. FeasOpt cannot analyze an infeasibility introduced by this parameter. If you want to analyze such a condition, add an explicit objective constraint to your model instead.

Default: 1.0e+75

datacheck (integer): controls data consistency checking and modeling assistance ↵

When the value of this parameter is set to level 2, CPLEX turns on both data consistency checking and modeling assistance. At this level, CPLEX issues warnings at the start of the optimization about disproportionate values (too large, too small) in coefficients, bounds, and righthand sides (RHS).

Default: 0

value	meaning
0	Data checking off
1	Data checking on
2	Data checking and model assistance on
depind (integer): dependency checker on/off ↵

This option determines if and when the dependency checker will be used.

Default: -1

value	meaning
-1	Automatic
0	Turn off dependency checking
1	Turn on only at the beginning of preprocessing
2	Turn on only at the end of preprocessing
3	Turn on at the beginning and at the end of preprocessing
dettilim (real): deterministic time limit ↵

Sets a time limit expressed in ticks, a unit to measure work done deterministically.

The length of a deterministic tick may vary by platform. Nevertheless, ticks are normally consistent measures for a given platform (combination of hardware and software) carrying the same load. In other words, the correspondence of ticks to clock time depends on the hardware, software, and the current load of the machine. For the same platform and same load, the ratio of ticks per second stays roughly constant, independent of the model solved. However, for very short optimization runs, the variation of this ratio is typically high.

Default: 1.0e+75

disjcuts (integer): disjunctive cuts generation ↵

Determines whether or not to generate disjunctive cuts during optimization. At the default of 0, generation is continued only if it seems to be helping.

Default: 0

value	meaning
-1	Do not generate disjunctive cuts
0	Determined automatically
1	Generate disjunctive cuts moderately
2	Generate disjunctive cuts aggressively
3	Generate disjunctive cuts very aggressively
divetype (integer): MIP dive strategy ↵

The MIP traversal strategy occasionally performs probing dives, where it looks ahead at both children nodes before deciding which node to choose. The default (automatic) setting chooses when to perform a probing dive, and the other two settings direct Cplex when to perform probing dives: never or always.

Default: 0

value	meaning
0	Automatic
1	Traditional dive
2	Probing dive
3	Guided dive
.divflt (real): solution pool range filter coefficients ↵

A diversity filter for a solution pool (see option SolnPool) allows you generate solutions that are similar to (or different from) a set of reference values that you specify for a set of binary variables. In particular, you can use a diversity filter to generate more solutions that are similar to an existing solution or to an existing partial solution.

A diversity filter drives the search for multiple solutions toward new solutions that satisfy a measure of diversity specified in the filter. This diversity measure applies only to binary variables. Potential new solutions are compared to a reference set. This reference set is specified with this dot option. If no reference set is specified, the difference measure will be computed relative to the other solutions in the pool. The diversity measure is computed by summing the pair-wise absolute differences from solution and the reference values.

Default: 0

divfltlo (real): lower bound on diversity ↵

Please check option DivFlt for general information on a diversity filter.

If you specify a lower bound on the diversity using DivFltLo, Cplex will look for solutions that are different from the reference values. In other words, you can say, Give me solutions that differ by at least this amount in this set of variables.

Default: mindouble

divfltup (real): upper bound on diversity ↵

Please check option DivFlt for general information on a diversity filter.

If you specify an upper bound on diversity DivFltUp, Cplex will look for solutions similar to the reference values. In other words, you can say, Give me solutions that are close to this one, within this set of variables.

Default: maxdouble

dpriind (integer): dual simplex pricing ↵

Pricing strategy for dual simplex method. Consider using dual steepest-edge pricing. Dual steepest-edge is particularly efficient and does not carry as much computational burden as the primal steepest-edge pricing.

Default: 0

value	meaning
0	Determined automatically
1	Standard dual pricing
2	Steepest-edge pricing
3	Steepest-edge pricing in slack space
4	Steepest-edge pricing, unit initial norms
5	Devex pricing
dynamicrows (integer): switch for dynamic management of rows ↵

This parameter specifies how CPLEX should manage rows in the current model during dual simplex optimization. More specifically, this parameter controls the use of the kernel simplex method (KSM) for the dual simplex algorithm. That is, CPLEX dynamically adjusts the dimensions of the basis matrix during execution of the dual simplex algorithm, according to the settings of this parameter.

When the value of this parameter is -1, its default value, this parameter specifies that the user wants CPLEX to manage rows dynamically, adjusting the dimensions of the basis matrix during dual simplex optimization. When it is set to 0, this parameter specifies that CPLEX must keep all rows. When it is set to 1, this parameter specifies that CPLEX can keep or discard rows according to its internal calculations.

Default: -1

value	meaning
-1	Automatic
0	Keep all rows
1	Manage rows
eachcutlim (integer): sets a limit for each type of cut ↵

This parameter allows you to set a uniform limit on the number of cuts of each type that Cplex generates. By default, the limit is a large integer; that is, there is no effective limit by default.

Tighter limits on the number of cuts of each type may benefit certain models. For example, a limit on each type of cut will prevent any one type of cut from being created in such large number that the limit on the total number of all types of cuts is reached before other types of cuts have an opportunity to be created. A setting of 0 means no cuts.

This parameter does not influence the number of Gomory cuts. For means to control the number of Gomory cuts, see also the fractional cut parameters: FracCand, FracCuts, and FracPass.

Default: 2100000000

epagap (real): absolute stopping tolerance ↵

Synonym: optca

Absolute tolerance on the gap between the best integer objective and the objective of the best node remaining. When the value falls below the value of the epagap setting, the optimization is stopped. This option overrides GAMS OptCA which provides its initial value.

Note: This option also influences the SubMIPs (e.g., used for the RINS heuristic) and can thus influence the solution path.

Default: GAMS OptCA

epgap (real): relative stopping tolerance ↵

Synonym: optcr

Relative tolerance on the gap between the best integer objective and the objective of the best node remaining. When the value falls below the value of the epgap setting, the mixed integer optimization is stopped. Note the difference in the Cplex definition of the relative tolerance with the GAMS definition. This option overrides GAMS OptCR which provides its initial value.

Note: This option also influences the SubMIPs (e.g., used for the RINS heuristic) and can thus influence the solution path.

Range: [0.0, 1.0]

Default: GAMS OptCR

epint (real): integrality tolerance ↵

Integrality Tolerance. This specifies the amount by which an integer variable can be different than an integer and still be considered feasible.

Range: [0.0, 0.5]

Default: 1.0e-05

eplin (real): degree of tolerance used in linearization ↵

Default: 0.001

epmrk (real): Markowitz pivot tolerance ↵

The Markowitz tolerance influences pivot selection during basis factorization. Increasing the Markowitz threshold may improve the numerical properties of the solution.

Range: [1.0e-04, 1.0]

Default: 0.01

epopt (real): optimality tolerance ↵

The optimality tolerance influences the reduced-cost tolerance for optimality. This option setting governs how closely Cplex must approach the theoretically optimal solution.

Range: [1.0e-09, 0.1]

Default: 1.0e-06

epper (real): perturbation constant ↵

Perturbation setting. Highly degenerate problems tend to stall optimization progress. Cplex automatically perturbs the variable bounds when this occurs. Perturbation expands the bounds on every variable by a small amount thereby creating a different but closely related problem. Generally, the solution to the less constrained problem is easier to solve. Once the solution to the perturbed problem has advanced as far as it can go, Cplex removes the perturbation by resetting the bounds to their original values.

If the problem is perturbed more than once, the perturbation constant is probably too large. Reduce the epper option to a level where only one perturbation is required. Any value greater than or equal to 1.0e-8 is valid.

Default: 1.0e-06

eprhs (real): feasibility tolerance ↵

Feasibility tolerance. This specifies the degree to which a problem's basic variables may violate their bounds. This tolerance influences the selection of an optimal basis and can be reset to a higher value when a problem is having difficulty maintaining feasibility during optimization. You may also wish to lower this tolerance after finding an optimal solution if there is any doubt that the solution is truly optimal. If the feasibility tolerance is set too low, Cplex may falsely conclude that a problem is infeasible.

Range: [1.0e-09, 0.1]

Default: 1.0e-06

exactkappa (boolean): report exact condition number in quality report ↵

Default: 0

feasopt (boolean): computes a minimum-cost relaxation to make an infeasible model feasible ↵

With Feasopt turned on, a minimum-cost relaxation of the right hand side values of constraints or bounds on variables is computed in order to make an infeasible model feasible. It marks the relaxed right hand side values and bounds in the solution listing.

Several options are available for the metric used to determine what constitutes a minimum-cost relaxation which can be set by option FeasOptMode.

Feasible relaxations are available for all problem types with the exception of quadratically constraint problems.

Default: 0

value	meaning
0	Turns Feasible Relaxation off
1	Turns Feasible Relaxation on
feasoptmode (integer): mode of FeasOpt ↵

The parameter FeasOptMode allows different strategies in finding feasible relaxation in one or two phases. In its first phase, it attempts to minimize its relaxation of the infeasible model. That is, it attempts to find a feasible solution that requires minimal change. In its second phase, it finds an optimal solution (using the original objective) among those that require only as much relaxation as it found necessary in the first phase. Values of the parameter FeasOptMode indicate two aspects: (1) whether to stop in phase one or continue to phase two and (2) how to measure the minimality of the relaxation (as a sum of required relaxations; as the number of constraints and bounds required to be relaxed; as a sum of the squares of required relaxations).

Default: 0

value	meaning
0	Minimize sum of relaxations
Minimize the sum of all required relaxations in first phase only
1	Minimize sum of relaxations and optimize
Minimize the sum of all required relaxations in first phase and execute second phase to find optimum among minimal relaxations
2	Minimize number of relaxations
Minimize the number of constraints and bounds requiring relaxation in first phase only
3	Minimize number of relaxations and optimize
Minimize the number of constraints and bounds requiring relaxation in first phase and execute second phase to find optimum among minimal relaxations
4	Minimize sum of squares of relaxations
Minimize the sum of squares of required relaxations in first phase only
5	Minimize sum of squares of relaxations and optimize
Minimize the sum of squares of required relaxations in first phase and execute second phase to find optimum among minimal relaxations
.feaspref (real): feasibility preference ↵

You can express the costs associated with relaxing a bound or right hand side value during a FeasOpt run through the .feaspref option. The input value denotes the users willingness to relax a constraint or bound. More precisely, the reciprocal of the specified value is used to weight the relaxation of that constraint or bound. The user may specify a preference value less than or equal to 0 (zero), which denotes that the corresponding constraint or bound must not be relaxed.

Default: 1

fixoptfile (string): name of option file which is read just before solving the fixed problem ↵

flowcovers (integer): flow cover cut generation ↵

Determines whether or not flow cover cuts should be generated during optimization.

Default: 0

value	meaning
-1	Do not generate flow cover cuts
0	Determined automatically
1	Generate flow cover cuts moderately
2	Generate flow cover cuts aggressively
flowpaths (integer): flow path cut generation ↵

Determines whether or not flow path cuts should be generated during optimization. At the default of 0, generation is continued only if it seems to be helping.

Default: 0

value	meaning
-1	Do not generate flow path cuts
0	Determined automatically
1	Generate flow path cuts moderately
2	Generate flow path cuts aggressively
folding (integer): LP folding will be attempted during the preprocessing phase ↵

Default: -1

value	meaning
-1	Automatic
0	Turn off folder
1	Moderate level of folding
2	Aggressive level of folding
3	Very aggressive level of folding
4	Highly aggressive level of folding
5	Extremely aggressive level of folding
fpheur (integer): feasibility pump heuristic ↵

Controls the use of the feasibility pump heuristic for mixed integer programming (MIP) models.

Default: 0

value	meaning
-1	Turns Feasible Pump heuristic off
0	Automatic
1	Apply the feasibility pump heuristic with an emphasis on finding a feasible solution
2	Apply the feasibility pump heuristic with an emphasis on finding a feasible solution with a good objective value
fraccand (integer): candidate limit for generating Gomory fractional cuts ↵

Limits the number of candidate variables for generating Gomory fractional cuts.

Default: 200

fraccuts (integer): Gomory fractional cut generation ↵

Determines whether or not Gomory fractional cuts should be generated during optimization.

Default: 0

value	meaning
-1	Do not generate Gomory fractional cuts
0	Determined automatically
1	Generate Gomory fractional cuts moderately
2	Generate Gomory fractional cuts aggressively
fracpass (integer): maximum number of passes for generating Gomory fractional cuts ↵

Sets the upper limit on the number of passes that will be performed when generating Gomory fractional cuts on a mixed integer model. Ignored if parameter FracCuts is set to a nonzero value.

Default: 0

value	meaning
0	0 Automatically determined
>0	Maximum passes to perform
freegamsmodel (boolean): preserves memory by dumping the GAMS model instance representation temporarily to disk ↵

In order to provide the maximum amount of memory to the solver this option dumps the internal representation of the model instance temporarily to disk and frees memory. This option only works with SolveLink=0 and only for models without quadratic constraints.

Default: 0

gubcovers (integer): GUB cover cut generation ↵

Determines whether or not GUB (Generalized Upper Bound) cover cuts should be generated during optimization. The default of 0 indicates that the attempt to generate GUB cuts should continue only if it seems to be helping.

Default: 0

value	meaning
-1	Do not generate GUB cover cuts
0	Determined automatically
1	Generate GUB cover cuts moderately
2	Generate GUB cover cuts aggressively
heurfreq (integer): heuristic frequency ↵

This option specifies how often to apply the node heuristic. Setting to a positive number applies the heuristic at the requested node interval. A value of 100, for example, means that heuristics are invoked every hundredth node in the tree.

Default: 0

value	meaning
-1	Do not use the node heuristic
0	Determined automatically
>0	Call heuristic at the requested node interval
heuristiceffort (real): the effort that CPLEX spends on heuristics during a MIP solve ↵

The value is used to increase (if >1) or decrease (if <1) the effort that CPLEX spends on heuristics during a MIP solve. If set to 0, no heuristic will run.

Default: 1.0

iafile (string): secondary option file to be read in interactive mode triggered by iatriggerfile ↵

If in interactive mode and this option is set, options will be read from the file specified by this option instead of direct user input (as described in interactive). This option file read can be triggered by interrupting Cplex with a Control-C or using the option iatriggerfile. If defined, GAMS/CPLEX looks for this file (content irrelevant) all iatriggertime seconds and if found, reads the option file iafile. The iatriggerfile is removed afterwards so it does not trigger twice.

iatriggerfile (string): file that triggers the reading of a secondary option file in interactive mode ↵

See iafile.

iatriggertime (real): time interval in seconds the link looks for the trigger file in interactive mode ↵

See iafile.

Default: 60

iis (integer): run the conflict refiner also known as IIS finder if the problem is infeasible ↵

Find an set of conflicting constraints or IIS (Irreducably Inconsistent Set) and write an conflict report to the GAMS solution listing if the model is found to be infeasible.

Default: 0

value	meaning
0	No conflict analysis
1	Conflict analysis after solve if infeasible
2	Conflict analysis without previous solve
implbd (integer): implied bound cut generation ↵

Determines whether or not implied bound cuts should be generated during optimization.

Default: 0

value	meaning
-1	Do not generate implied bound cuts
0	Determined automatically
1	Generate implied bound cuts moderately
2	Generate implied bound cuts aggressively
indicoptstrict (boolean): abort in case of an error in indicator constraint in solver option file ↵

If enabled and a variable or equation specified in an indicator constraint is not present in the model, model generation will abort with an error message. Otherwise, if this option is disabled, erroneous indicator constraints are ignored and a warning is printed.

Default: 1

interactive (boolean): allow interactive option setting after a Control-C ↵

When set to yes, options can be set interactively after interrupting Cplex with a Control-C. Options are entered just as if they were being entered in the cplex.opt file. Control is returned to Cplex by entering continue. The optimization can be aborted by entering abort. This option can only be used when running from the command line. Moreover, the GAMS option InteractiveSolver needs to be set to 1.

Default: 0

intsollim (integer): maximum number of integer solutions ↵

This option limits the MIP optimization to finding only this number of mixed integer solutions before stopping.

Default: large

itlim (integer): iteration limit ↵

Synonym: iterlim

The iteration limit option sets the maximum number of iterations before the algorithm terminates, without reaching optimality. This Cplex option overrides the GAMS IterLim option. Any non-negative integer value is valid.

Default: GAMS IterLim

.lazy (boolean): Lazy constraints activation ↵

Determines whether a linear constraint is treated as a lazy constraint. At the beginning of the MIP solution process, any constraint whose Lazy attribute is set to 1 (the default value is 0) is removed from the model and placed in the lazy constraint pool. Lazy constraints remain inactive until a feasible solution is found, at which point the solution is checked against the lazy constraint pool. If the solution violates any lazy constraint, the solution is discarded and one or more of the violated lazy constraints are pulled into the active model.

Lazy constraints are only active if option LazyConstraints is enabled and are specified through the option .lazy. The syntax for dot options is explained in the Introduction chapter of the Solver Manual.

Default: 0

lazyconstraints (boolean): Indicator to use lazy constraints ↵

Default: 0

lbheur (boolean): local branching heuristic ↵

This parameter lets you control whether Cplex applies a local branching heuristic to try to improve new incumbents found during a MIP search. By default, this parameter is off. If you turn it on, Cplex will invoke a local branching heuristic only when it finds a new incumbent. If Cplex finds multiple incumbents at a single node, the local branching heuristic will be applied only to the last one found.

Default: 0

value	meaning
0	Off
1	Apply local branching heuristic to new incumbent
liftprojcuts (integer): lift-and-project cuts ↵

Default: 0

value	meaning
-1	Do not generate lift-and-project cuts
0	Determined automatically
1	Generate lift-and-project cuts moderately
2	Generate lift-and-project cuts aggressively
3	Generate lift-and-project cuts very aggressively
localimplied (integer): generation of locally valid implied bound cuts ↵

Default: 0

value	meaning
-1	Do not generate locally valid implied bound cuts
0	Determined automatically
1	Generate locally valid implied bound cuts moderately
2	Generate locally valid implied bound cuts aggressively
3	Generate locally valid implied bound cuts very aggressively
lowerobjstop (real): in a minimization MILP or MIQP, the solver will abort the optimization process as soon as it finds a solution of value lower than or equal to the specified value ↵

Default: -1e75

lpmethod (integer): algorithm to be used for LP problems ↵

The default setting means that CPLEX will select the algorithm in a way that should give best overall performance.

For specific problem classes, the following details document the automatic settings. Note that future versions of CPLEX could adopt different strategies. Therefore, if you select any nondefault settings, you should review them periodically.

Currently, the behavior of the automatic setting is that CPLEX almost always invokes the dual simplex algorithm when it is solving an LP model from scratch. When it is continuing from an advanced basis, it will check whether the basis is primal or dual feasible, and choose the primal or dual simplex algorithm accordingly.

If multiple threads have been requested (see threads), in either deterministic or opportunistic mode, the concurrent optimization algorithm is selected by the automatic setting when CPLEX is solving a continuous linear programming model (LP) from scratch.

When three or more threads are available, and you select concurrent optimization for the value of this parameter, its behavior depends on whether parallel mode is opportunistic or deterministic (default parallel mode). Concurrent optimization in opportunistic parallel mode runs the dual simplex optimizer on one thread, the primal simplex optimizer on a second thread, the parallel barrier optimizer on a third thread and on any additional available threads. In contrast, concurrent optimization in deterministic parallel mode runs the dual optimizer on one thread and the parallel barrier optimizer on any additional available threads.

The automatic setting may be expanded in the future so that CPLEX chooses the algorithm based on additional problem characteristics.

Default: 0

value	meaning
0	Automatic
1	Primal Simplex
2	Dual Simplex
3	Network Simplex
4	Barrier
5	Sifting
6	Concurrent
ltol (real): basis identification primal tolerance ↵

Default: 0

mcfcuts (integer): multi-commodity flow cut generation ↵

Specifies whether Cplex should generate multi-commodity flow (MCF) cuts in a problem where Cplex detects the characteristics of a multi-commodity flow network with arc capacities. By default, Cplex decides whether or not to generate such cuts. To turn off generation of such cuts, set this parameter to -1. Cplex is able to recognize the structure of a network as represented in many real-world models. When it recognizes such a network structure, Cplex is able to generate cutting planes that usually help solve such problems. In this case, the cuts that Cplex generates state that the capacities installed on arcs pointing into a component of the network must be at least as large as the total flow demand of the component that cannot be satisfied by flow sources within the component.

Default: 0

value	meaning
-1	Do not generate MCF cuts
0	Determined automatically
1	Generate MCF cuts moderately
2	Generate MCF cuts aggressively
memoryemphasis (boolean): reduces use of memory ↵

This parameter lets you indicate to Cplex that it should conserve memory where possible. When you set this parameter to its non default value, Cplex will choose tactics, such as data compression or disk storage, for some of the data computed by the barrier and MIP optimizers. Of course, conserving memory may impact performance in some models. Also, while solution information will be available after optimization, certain computations that require a basis that has been factored (for example, for the computation of the condition number Kappa) may be unavailable.

Default: 0

value	meaning
0	Do not conserve memory
1	Conserve memory where possible
mipdisplay (integer): progress display level ↵

The amount of information displayed during MIP solution increases with increasing values of this option.

Default: 4

value	meaning
0	No display
1	Display integer feasible solutions
2	Displays nodes under mipinterval control
3	Same as 2 but adds information on cuts
4	Same as 3 but adds LP display for the root node
5	Same as 3 but adds LP display for all nodes
mipemphasis (integer): MIP solution tactics ↵

This option controls the tactics for solving a mixed integer programming problem.

Default: 0

value	meaning
0	Balance optimality and feasibility
1	Emphasize feasibility over optimality
2	Emphasize optimality over feasibility
3	Emphasize moving the best bound
4	Emphasize hidden feasible solutions
5	Find high quality feasible solutions as early as possible
mipinterval (integer): progress display interval ↵

Controls the frequency of node logging when the parameter MIPDisplay is set higher than 1 (one). Frequency must be an integer; it may be 0 (zero), positive, or negative. By default, CPLEX displays new information in the node log during a MIP solve at relatively high frequency during the early stages of solving a MIP model, and adds lines to the log at progressively longer intervals as solving continues. In other words, CPLEX logs information frequently in the beginning and progressively less often as it works. When the value is a positive integer n, CPLEX displays new incumbents, plus it displays a new line in the log every n nodes. When the value is a negative integer n, CPLEX displays new incumbents, and the negative value determines how much processing CPLEX does before it displays a new line in the node log. A negative value close to zero means that CPLEX displays new lines in the log frequently. A negative value far from zero means that CPLEX displays new lines in the log less frequently. In other words, a negative value of this parameter contracts or dilates the interval at which CPLEX displays information in the node log.

Default: 0

mipkappastats (integer): MIP kappa computation ↵

MIP kappa summarizes the distribution of the condition number of the optimal bases CPLEX encountered during the solution of a MIP model. That summary may let you know more about the numerical difficulties of your MIP model. Because MIP kappa (as a statistical distribution) requires CPLEX to compute the condition number of the optimal bases of the subproblems during branch-and-cut search, you can compute the MIP kappa only when CPLEX solves the subproblem with its simplex optimizer. In other words, in order to obtain results with this parameter, you can not use the sifting optimizer nor the barrier without crossover to solve the subproblems. See the parameters StartAlg and SubAlg.

Computing the kappa of a subproblem has a cost. In fact, computing MIP kappa for the basis matrices can be computationally expensive and thus generally slows down the solution of a problem. Therefore, the setting 0 (automatic) tells CPLEX generally not to compute MIP kappa, but in cases where the parameter NumericalEmphasis is turned on, CPLEX computes MIP kappa for a sample of subproblems. The value 1 (sample) leads to a negligible performance degradation on average, but can slow down the branch-and-cut exploration by as much as 10% on certain models. The value 2 (full) leads to a 2% performance degradation on average, but can significantly slow the branch-and-cut exploration on certain models. In practice, the value 1 (sample) is a good trade-off between performance and accuracy of statistics. If you need very accurate statistics, then use value 2 (full).

In case CPLEX is instructed to compute a MIP kappa distribution, the parameter Quality is automatically turned on.

Default: 0

value	meaning
-1	No MIP kappa statistics; default
0	Automatic: let CPLEX decide
1	Compute MIP kappa for a sample of subproblems
2	Compute MIP kappa for all subproblems
mipordind (boolean): priority list on/off ↵

Synonym: prioropt

Use priorities. Priorities should be assigned based on your knowledge of the problem. Variables with higher priorities will be branched upon before variables of lower priorities. This direction of the tree search can often dramatically reduce the number of nodes searched. For example, consider a problem with a binary variable representing a yes/no decision to build a factory, and other binary variables representing equipment selections within that factory. You would naturally want to explore whether or not the factory should be built before considering what specific equipment to purchased within the factory. By assigning a higher priority to the build/no build decision variable, you can force this logic into the tree search and eliminate wasted computation time exploring uninteresting portions of the tree. When set at 0 (default), the MIPOrdInd option instructs Cplex not to use priorities for branching. When set to 1, priority orders are utilized.

Note: Priorities are assigned to discrete variables using the .prior suffix in the GAMS model. Lower .prior values mean higher priority. The .prioropt model suffix has to be used to signal GAMS to export the priorities to the solver.

Default: GAMS PriorOpt

value	meaning
0	Do not use priorities for branching
1	Priority orders are utilized
mipordtype (integer): priority order generation ↵

This option is used to select the type of generic priority order to generate when no priority order is present.

Default: 0

value	meaning
0	None
1	decreasing cost magnitude
2	increasing bound range
3	increasing cost per coefficient count
mipsearch (integer): search strategy for mixed integer programs ↵

Sets the search strategy for a mixed integer program. By default, Cplex chooses whether to apply dynamic search or conventional branch and cut based on characteristics of the model.

Default: 0

value	meaning
0	Automatic
1	Apply traditional branch and cut strategy
2	Apply dynamic search
mipstart (integer): use mip starting values ↵

This option controls the use of advanced starting values for mixed integer programs. A setting of 2 indicates that the values should be checked to see if they provide an integer feasible solution before starting optimization. For mipstart equals 1, 2, 3 or 4 fractional values are rounded to the nearest integer value if the integrality violation is larger than CPLEX's integer tolerance and smaller or equal to tryint. A partial MIP start is applied for mipstart equals 1, 3 or 4. Here, for discrete variables only integer values (after possible rounding) are added to the advanced starting values.

Default: 0

value	meaning
0	do not use the values
1	set discrete variable values and use auto mipstart level
2	set all variable values and use check feasibility mipstart level
3	set discrete variable values and use solve fixed mipstart level
4	set discrete variable values and use solve sub-MIP mipstart level
5	set discrete variable values and use solve repair-MIP mipstart level
6	set discrete variable values and use no checks at all. Warning: CPLEX may accept infeasible points as solutions!
mipstopexpr (string): stopping expression for branch and bound ↵

If the provided logical expression is true, the branch-and-bound is aborted. Supported values are: etalg, resusd, nodusd, objest, objval, is_feasible. Supported opertators are: +, -, *, /, ^, %, !=, ==, <, <=, >, >=, !, &&, ||, (, ), abs, ceil, exp, floor, log, log10, pow, sqrt. Example:

nodusd >= 1000 && is_feasible && abs(objest - objval) / abs(objval) < 0.1
If multiple stop expressions are given in an option file, the algorithm stops if any of them is true (|| concatenation).

miptrace (string): filename of MIP trace file ↵

For a description of this feature, see chapter Solve trace.

Note: In contrast to other solvers, GAMS/CPLEX doesn't append the MIP trace file after a certain time or node count, but when CPLEX reports global progress. In order to indicate this, the MIP trace file will show X instead of N or T.

miqcpstrat (integer): MIQCP relaxation choice ↵

This option controls how MIQCPs are solved. For some models, the setting 2 may be more effective than 1. You may need to experiment with this parameter to determine the best setting for your model.

Default: 0

value	meaning
0	Automatic
1	QCP relaxation
Cplex will solve a QCP relaxation of the model at each node.
2	LP relaxation
Cplex will solve a LP relaxation of the model at each node.
mircuts (integer): mixed integer rounding cut generation ↵

Determines whether or not to generate mixed integer rounding (MIR) cuts during optimization. At the default of 0, generation is continued only if it seems to be helping.

Default: 0

value	meaning
-1	Do not generate MIR cuts
0	Determined automatically
1	Generate MIR cuts moderately
2	Generate MIR cuts aggressively
mpslongnum (boolean): MPS file format precision of numeric output ↵

Determines the precision of numeric output in the MPS file formats. When this parameter is set to its default value 1 (one), numbers are written to MPS files in full-precision; that is, up to 15 significant digits may be written. The setting 0 (zero) writes files that correspond to the standard MPS format, where at most 12 characters can be used to represent a value. This limit may result in loss of precision.

Default: 1

value	meaning
0	Use limited MPS precision
1	Use full-precision
mtol (real): basis identification dual tolerance ↵

Default: 0

multimipstart (string): use multiple mipstarts provided via gdx files ↵

Specifies (multiple) GDX files with values for the variables. Each file is treated as one intial guess for the MIP start. These MIP starts are added in addition to the initial guess provided by the level attribute. A MIP start GDX file can be created, for example, by using the command line option savepoint.

multobj (boolean): controls the hierarchical optimization of multiple objectives ↵

Default: 0

multobjdisplay (integer): level of display during multiobjective optimization ↵

Default: 1

value	meaning
0	No display
1	Summary display after each subproblem
2	Summary display after each subproblem, as well as subproblem logs
multobjmethod (integer): method used for multi-objective solves ↵

When solving a continuous multi-objective model using a hierarchical approach, the model is solved once for each objective. The algorithm used to solve for the highest priority objective is controlled by the LPMethod parameter. This parameter determines the algorithm used to solve for subsequent objectives.

Default: 0

multobjoptfiles (string): List of option files used for individual solves within multi-objective optimization ↵

The options given by the option files in multobjoptfiles are applied on top of the default GAMS/CPLEX options. This includes options set by the user via the standard option file.

If the list of option files in multobjoptfiles is less than the number of objective functions, the default GAMS/CPLEX options (incl. user options as before) are used to solve the remaining instances. Additional option files (i.e. more than objective functions) are ignored.

Applied options can be verified by setting multobjdisplay to 2.

multobjtolmip (boolean): enables hard constraints for hierarchical optimization objectives based on degradation tolerances ↵

CPLEX supports two different strategies to handle the degradation tolerances objnabstol and objnreltol depending on the problem type (continuous or discrete), see objnabstol. This setting enables the discrete strategy for continous models. Note that objnreltol has no effect for discrete models. Enabling this option can lead to higher solution times.

Default: 1

names (boolean): load GAMS names into Cplex ↵

This option causes GAMS names for the variables and equations to be loaded into Cplex. These names will then be used for error messages, log entries, and so forth. Setting names to no may help if memory is very tight.

Default: 1

netdisplay (integer): network display level ↵

This option controls the log for network iterations.

Default: 2

value	meaning
0	No network log.
1	Displays true objective values
2	Displays penalized objective values
netepopt (real): optimality tolerance for the network simplex method ↵

This optimality tolerance influences the reduced-cost tolerance for optimality when using the network simplex method. This option setting governs how closely Cplex must approach the theoretically optimal solution.

Range: [1.0e-11, 0.1]

Default: 1.0e-06

neteprhs (real): feasibility tolerance for the network simplex method ↵

This feasibility tolerance determines the degree to which the network simplex algorithm will allow a flow value to violate its bounds.

Range: [1.0e-11, 0.1]

Default: 1.0e-06

netfind (integer): attempt network extraction ↵

Specifies the level of network extraction to be done.

Default: 2

value	meaning
1	Extract pure network only
2	Try reflection scaling
3	Try general scaling
netitlim (integer): iteration limit for network simplex ↵

Iteration limit for the network simplex method.

Default: large

netppriind (integer): network simplex pricing ↵

Network simplex pricing algorithm. The default of 0 (currently equivalent to 3) shows best performance for most problems.

Default: 0

value	meaning
0	Automatic
1	Partial pricing
2	Multiple partial pricing
3	Multiple partial pricing with sorting
nodecuts (integer): decide whether or not cutting planes are separated at the nodes of the branch-and-bound tree ↵

Default: 0

nodefileind (integer): node storage file indicator ↵

Specifies how node files are handled during MIP processing. Used when parameter WorkMem has been exceeded by the size of the branch and cut tree. If set to 0 when the tree memory limit is reached, optimization is terminated. Otherwise a group of nodes is removed from the in-memory set as needed. By default, Cplex transfers nodes to node files when the in-memory set is larger than 128 MBytes, and it keeps the resulting node files in compressed form in memory. At settings 2 and 3, the node files are transferred to disk. They are stored under a directory specified by parameter WorkDir and Cplex actively manages which nodes remain in memory for processing.

Default: 1

value	meaning
0	No node files
1	Node files in memory and compressed
2	Node files on disk
3	Node files on disk and compressed
nodelim (integer): maximum number of nodes to solve ↵

Synonym: nodlim

The maximum number of nodes solved before the algorithm terminates, without reaching optimality. This option overrides the GAMS NodLim model suffix. When this parameter is set to 0 (this is only possible through an option file), Cplex completes processing at the root; that is, it creates cuts and applies heuristics at the root. When this parameter is set to 1 (one), it allows branching from the root; that is, nodes are created but not solved.

Default: GAMS NodLim

nodesel (integer): node selection strategy ↵

This option is used to set the rule for selecting the next node to process when backtracking.

Default: 1

value	meaning
0	Depth-first search
This chooses the most recently created node.
1	Best-bound search
This chooses the unprocessed node with the best objective function for the associated LP relaxation.
2	Best-estimate search
This chooses the node with the best estimate of the integer objective value that would be obtained once all integer infeasibilities are removed.
3	Alternate best-estimate search
numericalemphasis (boolean): emphasizes precision in numerically unstable or difficult problems ↵

This parameter lets you indicate to Cplex that it should emphasize precision in numerically difficult or unstable problems, with consequent performance trade-offs in time and memory.

Default: 0

value	meaning
0	Off
1	Exercise extreme caution in computation
objdif (real): overrides GAMS Cheat parameter ↵

Synonym: cheat

A means for automatically updating the cutoff to more restrictive values. Normally the most recently found integer feasible solution objective value is used as the cutoff for subsequent nodes. When this option is set to a positive value, the value will be subtracted from (added to) the newly found integer objective value when minimizing (maximizing). This forces the MIP optimization to ignore integer solutions that are not at least this amount better than the one found so far. The option can be adjusted to improve problem solving efficiency by limiting the number of nodes; however, setting this option at a value other than zero (the default) can cause some integer solutions, including the true integer optimum, to be missed. Negative values for this option will result in some integer solutions that are worse than or the same as those previously generated, but will not necessarily result in the generation of all possible integer solutions. This option overrides the GAMS Cheat parameter.

Default: 0.0

objllim (real): objective function lower limit ↵

Setting a lower objective function limit will cause Cplex to halt the optimization process once the minimum objective function value limit has been exceeded.

Default: -1.0e+75

objnabstol (string): allowable absolute degradation for objective ↵

This parameter is used to set the allowable degradation for an objective when doing hierarchical multi-objective optimization (MultObj). The syntax for this parameter is ObjNAbsTol ObjVarName value.

Hierarchical multi-objective optimization will optimize for the different objectives in the model one at a time, in priority order. For MIPs (or if MultObjTolMip is enabled), if it achieves objective value z when it optimizes for this objective, then subsequent steps are allowed to degrade this value by at most ObjNAbsTol. For LPs, ObjNAbsTol defines a threshold for reduced costs above which nonbasic variables in the associated LP solve will be fixed at the bound at which they reside.

objnreltol (string): allowable relative degradation for objective ↵

This parameter is used to set the allowable degradation for an objective when doing hierarchical multi-objective optimization (MultObj). The syntax for this parameter is ObjNRelTol ObjVarName value.

Hierarchical multi-objective optimization will optimize for the different objectives in the model one at a time, in priority order. For MIPs (or if MultObjTolMip is enabled), if it achieves objective value z when it optimizes for this objective, then subsequent steps are allowed to degrade this value by at most ObjNRelTol*|z|. This option has no effect for continuous models.

objrng (string): do objective ranging ↵

Calculate sensitivity ranges for the specified GAMS variables. Unlike most options, ObjRng can be repeated multiple times in the options file. Sensitivity range information will be produced for each GAMS variable named. Specifying all will cause range information to be produced for all variables. Range information will be printed to the beginning of the solution listing in the GAMS listing file unless option RngRestart is specified.

Default: no objective ranging is done

objulim (real): objective function upper limit ↵

Setting an upper objective function limit will cause Cplex to halt the optimization process once the maximum objective function value limit has been exceeded.

Default: 1.0e+75

optimalitytarget (integer): type of optimality that Cplex targets ↵

This parameter specifies the type of solution that CPLEX attempts to compute with respect to the optimality of that solution when CPLEX solves a continuous (QP) or mixed integer (MIQP) quadratic model. In other words, the variables of the model can be continuous or mixed integer and continuous; the objective function includes a quadratic term, and perhaps the objective function is not positive semi-definite (non PSD). This parameter does not apply to quadratically constrained mixed integer problems (MIQCP); that is, this parameter does not apply to mixed integer problems that include a quadratic term among the constraints.

Default: 0

value	meaning
0	Automatic
CPLEX first attempts to compute a provably optimal solution. If CPLEX cannot compute a provably optimal solution because the objective function is not convex, CPLEX will return with an error (Q is not PSD).
1	Search for a globally optimal solution to a convex model
CPLEX searches for a globally optimal solution to a convex model. In problems of type QP or MIQP, this setting interacts with linearization switch QToLin for QP, MIQP
2	Search for a solution that satisfies first-order optimality conditions no optimality guarantee
CPLEX first attempt to compute a provably optimal solution. If CPLEX cannot compute a provably optimal solution because the objective function is not convex, CPLEX searches for a solution that satisfies first-order optimality conditions but is not necessarily globally optimal.
3	Search for a globally optimal solution regardless of convexity
If the problem type is QP, CPLEX first changes the problem type to MIQP. CPLEX then solves the problem (whether originally QP or MIQP) to global optimality. In problems of type QP or MIQP, this setting interacts with with linearization switch QToLin for QP, MIQP. With this setting information about dual values is not available for the solution.
parallelmode (integer): parallel optimization mode ↵

Sets the parallel optimization mode. Possible modes are automatic, deterministic, and opportunistic.

In this context, deterministic means that multiple runs with the same model at the same parameter settings on the same platform will reproduce the same solution path and results. In contrast, opportunistic implies that even slight differences in timing among threads or in the order in which tasks are executed in different threads may produce a different solution path and consequently different timings or different solution vectors during optimization executed in parallel threads. When running with multiple threads, the opportunistic setting entails less synchronization between threads and consequently may provide better performance.

In deterministic mode, Cplex applies as much parallelism as possible while still achieving deterministic results. That is, when you run the same model twice on the same platform with the same parameter settings, you will see the same solution and optimization run.

More opportunities to exploit parallelism are available if you do not require determinism. In other words, Cplex can find more opportunities for parallelism if you do not require an invariant, repeatable solution path and precisely the same solution vector. To use all available parallelism, you need to select the opportunistic parallel mode. In this mode, Cplex will utilize all opportunities for parallelism in order to achieve best performance.

However, in opportunistic mode, the actual optimization may differ from run to run, including the solution time itself and the path traveled in the search.

Parallel MIP optimization can be opportunistic or deterministic.

Parallel barrier optimization is only deterministic.

A GAMS/Cplex run will use deterministic mode unless explicitly specified.

If ParallelMode is explicitly set to 0 (automatic) the settings of this parallel mode parameter interact with settings of the Threads parameter. Let the result number of threads available to Cplex be n (note that negative values for the threads parameter are possible to exclude work on some cores).

Here is is list of possible value:

Default: 1

value	meaning
-1	Enable opportunistic parallel search mode
0	Automatic
1	Enable deterministic parallel search mode
perind (boolean): force initial perturbation ↵

Perturbation Indicator. If a problem automatically perturbs early in the solution process, consider starting the solution process with a perturbation by setting PerInd to 1. Manually perturbing the problem will save the time of first allowing the optimization to stall before activating the perturbation mechanism, but is useful only rarely, for extremely degenerate problems.

Default: 0

value	meaning
0	not automatically perturbed
1	automatically perturbed
perlim (integer): number of stalled iterations before perturbation ↵

Perturbation limit. The number of stalled iterations before perturbation is invoked. The default value of 0 means the number is determined automatically.

Default: 0

polishafterdettime (real): deterministic time before starting to polish a feasible solution ↵

Default: 1.0e+75

polishafterepagap (real): absolute MIP gap before starting to polish a feasible solution ↵

Solution polishing can yield better solutions in situations where good solutions are otherwise hard to find. More time-intensive than other heuristics, solution polishing is actually a variety of branch-and-cut that works after an initial solution is available. In fact, it requires a solution to be available for polishing, either a solution produced by branch-and-cut, or a MIP start supplied by a user. Because of the high cost entailed by solution polishing, it is not called throughout branch-and-cut like other heuristics. Instead, solution polishing works in a second phase after a first phase of conventional branch-and-cut. As an additional step after branch-and-cut, solution polishing can improve the best known solution. As a kind of branch-and-cut algorithm itself, solution polishing focuses solely on finding better solutions. Consequently, it may not prove optimality, even if the optimal solution has indeed been found. Like the RINS heuristic, solution polishing explores neighborhoods of previously found solutions by solving subMIPs.

Sets an absolute MIP gap (that is, the difference between the best integer objective and the objective of the best node remaining) after which CPLEX stops branch-and-cut and begins polishing a feasible solution. The default value is such that CPLEX does not invoke solution polishing by default.

Default: 0.0

polishafterepgap (real): relative MIP gap before starting to polish a solution ↵

Sets a relative MIP gap after which CPLEX will stop branch-and-cut and begin polishing a feasible solution. The default value is such that CPLEX does not invoke solution polishing by default.

Default: 0.0

polishafterintsol (integer): MIP integer solutions to find before starting to polish a feasible solution ↵

Sets the number of integer solutions to find before CPLEX stops branch-and-cut and begins to polish a feasible solution. The default value is such that CPLEX does not invoke solution polishing by default.

Default: 2147483647

polishafternode (integer): nodes to process before starting to polish a feasible solution ↵

Sets the number of nodes processed in branch-and-cut before CPLEX starts solution polishing, if a feasible solution is available.

Default: 2147483647

polishaftertime (real): time before starting to polish a feasible solution ↵

Tells CPLEX how much time in seconds to spend during mixed integer optimization before CPLEX starts polishing a feasible solution. The default value is such that CPLEX does not start solution polishing by default.

Default: 1.0e+75

populatelim (integer): limit of solutions generated for the solution pool by populate method ↵

Limits the number of solutions generated for the solution pool during each call to the populate procedure. Populate stops when it has generated PopulateLim solutions. A solution is counted if it is valid for all filters (see DivFlt and consistent with the relative and absolute pool gap parameters (see SolnPoolGap and SolnPoolAGap), and has not been rejected by the incumbent checking routine (see UserIncbCall), whether or not it improves the objective of the model. This parameter does not apply to MIP optimization generally; it applies only to the populate procedure.

If you are looking for a parameter to control the number of solutions stored in the solution pool, consider the parameter SolnPoolCapacity instead.

Populate will stop before it reaches the limit set by this parameter if it reaches another limit, such as a time or node limit set by the user.

Default: 20

ppriind (integer): primal simplex pricing ↵

Pricing algorithm. Likely to show the biggest impact on performance. Look at overall solution time and the number of Phase I and total iterations as a guide in selecting alternate pricing algorithms. If you are using the dual Simplex method use DPriInd to select a pricing algorithm. If the number of iterations required to solve your problem is approximately the same as the number of rows in your problem, then you are doing well. Iteration counts more than three times greater than the number of rows suggest that improvements might be possible.

Default: 0

value	meaning
-1	Reduced-cost pricing
This is less compute intensive and may be preferred if the problem is small or easy. This option may also be advantageous for dense problems (say 20 to 30 nonzeros per column).
0	Hybrid reduced-cost and Devex pricing
1	Devex pricing
This may be useful for more difficult problems which take many iterations to complete Phase I. Each iteration may consume more time, but the reduced number of total iterations may lead to an overall reduction in time. Tenfold iteration count reductions leading to threefold speed improvements have been observed. Do not use devex pricing if the problem has many columns and relatively few rows. The number of calculations required per iteration will usually be disadvantageous.
2	Steepest edge pricing
If devex pricing helps, this option may be beneficial. Steepest-edge pricing is computationally expensive, but may produce the best results on exceptionally difficult problems.
3	Steepest edge pricing with slack initial norms
This reduces the computationally intensive nature of steepest edge pricing.
4	Full pricing
predual (integer): give dual problem to the optimizer ↵

Solve the dual. Some linear programs with many more rows than columns may be solved faster by explicitly solving the dual. The PreDual option will cause Cplex to solve the dual while returning the solution in the context of the original problem. This option is ignored if presolve is turned off.

Default: 0

value	meaning
-1	do not give dual to optimizer
0	automatic
1	give dual to optimizer
preind (boolean): turn presolver on/off ↵

Perform Presolve. This helps most problems by simplifying, reducing and eliminating redundancies. However, if there are no redundancies or opportunities for simplification in the model, if may be faster to turn presolve off to avoid this step. On rare occasions, the presolved model, although smaller, may be more difficult than the original problem. In this case turning the presolve off leads to better performance. Specifying 0 turns the aggregator off as well.

Default: 1

prepass (integer): number of presolve applications to perform ↵

Number of MIP presolve applications to perform. By default, Cplex determines this automatically. Specifying 0 turns off the presolve but not the aggregator. Set PreInd to 0 to turn both off.

Default: -1

value	meaning
-1	Determined automatically
0	No presolve
>0	Number of MIP presolve applications to perform
prereform (integer): set presolve reformulations ↵

Default: 3

preslvnd (integer): node presolve selector ↵

Indicates whether node presolve should be performed at the nodes of a mixed integer programming solution. Node presolve can significantly reduce solution time for some models. The default setting is generally effective.

Default: 0

value	meaning
-1	No node presolve
0	Automatic
1	Force node presolve
2	Perform probing on integer-infeasible variables
3	Perform aggressive node probing
pricelim (integer): pricing candidate list ↵

Size for the pricing candidate list. Cplex dynamically determines a good value based on problem dimensions. Only very rarely will setting this option manually improve performance. Any non-negative integer values are valid.

Default: 0, in which case it is determined automatically

printoptions (boolean): list values of all options to GAMS listing file ↵

Write the values of all options to the GAMS listing file. Valid values are no or yes.

Default: 0

probe (integer): perform probing before solving a MIP ↵

Determines the amount of probing performed on a MIP. Probing can be both very powerful and very time consuming. Setting the value to 1 can result in dramatic reductions or dramatic increases in solution time depending on the particular model.

Default: 0

value	meaning
-1	No probing
0	Automatic
1	Limited probing
2	More probing
3	Full probing
probedettime (real): deterministic time spent probing ↵

Default: 1.0e+75

probetime (real): time spent probing ↵

Limits the amount of time in seconds spent probing.

Default: 1.0e+75

qextractalg (integer): quadratic extraction algorithm in GAMS interface ↵

Default: 0

value	meaning
0	Automatic
1	ThreePass: Uses a three-pass forward / backward / forward AD technique to compute function / gradient / Hessian values and a hybrid scheme for storage.
2	DoubleForward: Uses forward-mode AD to compute and store function, gradient, and Hessian values at each node or stack level as required. The gradients and Hessians are stored in linked lists.
3	Concurrent: Uses ThreePass and DoubleForward in parallel. As soon as one finishes, the other one stops.
qpmakepsdind (boolean): adjust MIQP formulation to make the quadratic matrix positive-semi-definite ↵

Determines whether Cplex will attempt to adjust a MIQP formulation, in which all the variables appearing in the quadratic term are binary. When this feature is active, adjustments will be made to the elements of a quadratic matrix that is not nominally positive semi-definite (PSD, as required by Cplex for all QP formulations), to make it PSD, and will also attempt to tighten an already PSD matrix for better numerical behavior. The default setting of 1 means yes but you can turn it off if necessary; most models should benefit from the default setting.

Default: 1

value	meaning
0	Off
1	On
qpmethod (integer): algorithm to be used for QP problems ↵

Specifies which QP algorithm to use.

At the default of 0 (automatic), barrier is used for QP problems and dual simplex for the root relaxation of MIQP problems.

Default: 0

value	meaning
0	Automatic
1	Primal Simplex
2	Dual Simplex
3	Network Simplex
4	Barrier
5	Sifting
6	Concurrent dual, barrier, and primal
qtolin (integer): linearization of the quadratic terms in the objective function of a QP or MIQP model ↵

This parameter switches on or off linearization of the quadratic terms in the objective function of a quadratic program or of a mixed integer quadratic program.

In a convex mixed integer quadratic program, this parameter controls whether Cplex linearizes the product of binary variables in the objective function during presolve. In a nonconvex quadratic program or mixed integer quadratic program solved to global optimality according to OptimalityTarget, this parameter controls how Cplex linearizes the product of bounded variables in the objective function during presolve.

This parameter interacts with the existing parameter OptimalityTarget: When the solution target type is set to 1 (that is, Cplex searches for a globally optimal solution to a convex model), then in a convex MIQP, this parameter tells Cplex to replace the product of a binary variable and a bounded linear variable by a linearly constrained variable. When the solution target type is set to 3, then in a nonconvex QP or nonconvex MIQP, this parameter controls the initial relaxation.

Default: -1

value	meaning
-1	Automatic
0	Off, Cplex does not linearize quadratic terms in the objective
1	On, Cplex linearizes quadratic terms in the objective
quality (boolean): write solution quality statistics ↵

Write solution quality statistics to the listing and log file. If set to yes, the statistics appear after the Solve Summary and before the Solution Listing and contain information about infeasibility levels, solution value magnitued, and the condition number (kappa):

Solution Quality Statistics:
                                   unscaled                scaled
                               max         sum         max         sum
primal infeasibility        0.000e+00   0.000e+00   0.000e+00   0.000e+00
dual infeasibility          0.000e+00   0.000e+00   0.000e+00   0.000e+00
primal residual             0.000e+00   0.000e+00   0.000e+00   0.000e+00
dual residual               0.000e+00   0.000e+00   0.000e+00   0.000e+00
primal solution vector      3.000e+02   9.000e+02   3.000e+02   9.000e+02
dual solution vector        1.000e+00   1.504e+00   1.000e+00   1.504e+00
slacks                      5.000e+01   5.000e+01   5.000e+01   5.000e+01
reduced costs               3.600e-02   4.500e-02   3.600e-02   4.500e-02

Condition number of the scaled basis matrix =    9.000e+00
Default: 0

randomseed (integer): sets the random seed differently for diversity of solutions ↵

Default: changes with each Cplex release

readflt (string): reads Cplex solution pool filter file ↵

The GAMS/Cplex solution pool options cover the basic use of diversity and range filters for producing multiple solutions. If you need multiple filters, weights on diversity filters or other advanced uses of solution pool filters, you could produce a Cplex filter file with your favorite editor or the GAMS Put Facility and read this into GAMS/Cplex using this option.

readparams (string): read Cplex parameter file ↵

reduce (integer): primal and dual reduction type ↵

Determines whether primal reductions, dual reductions, or both, are performed during preprocessing. It is occasionally advisable to do only one or the other when diagnosing infeasible or unbounded models.

Default: 3

value	meaning
0	No primal or dual reductions
1	Only primal reductions
2	Only dual reductions
3	Both primal and dual reductions
reinv (integer): refactorization frequency ↵

Refactorization Frequency. This option determines the number of iterations between refactorizations of the basis matrix. The default should be optimal for most problems. Cplex's performance is relatively insensitive to changes in refactorization frequency. Only for extremely large, difficult problems should reducing the number of iterations between refactorizations be considered. Any non-negative integer value is valid.

Default: 0, in which case it is determined automatically

relaxfixedinfeas (boolean): accept small infeasibilties in the solve of the fixed problem ↵

Sometimes the solution of the fixed problem of a MIP does not solve to optimality due to small (dual) infeasibilities. The default behavior of the GAMS/Cplex link is to return the primal solution values only. If the option is set to 1, the small infeasibilities are ignored and a full solution including the dual values are reported back to GAMS.

Default: 0

value	meaning
0	Off
1	On
relaxpreind (integer): presolve for initial relaxation on/off ↵

This option will cause the Cplex presolve to be invoked for the initial relaxation of a mixed integer program (according to the other presolve option settings). Sometimes, additional reductions can be made beyond any MIP presolve reductions that may already have been done.

Default: -1

value	meaning
-1	Automatic
0	do not presolve initial relaxation
1	use presolve on initial relaxation
relobjdif (real): relative cheat parameter ↵

The relative version of the ObjDif option. Ignored if ObjDif is non-zero.

Default: 0.0

repairtries (integer): try to repair infeasible MIP start ↵

This parameter lets you indicate to Cplex whether and how many times it should try to repair an infeasible MIP start that you supplied. The parameter has no effect if the MIP start you supplied is feasible. It has no effect if no MIP start was supplied.

Default: 0

value	meaning
-1	None: do not try to repair
0	Automatic
>0	Maximum tries to perform
repeatpresolve (integer): reapply presolve at root after preprocessing ↵

This integer parameter tells Cplex whether to re-apply presolve, with or without cuts, to a MIP model after processing at the root is otherwise complete.

Default: -1

value	meaning
-1	Automatic
0	Turn off represolve
1	Represolve without cuts
2	Represolve with cuts
3	Represolve with cuts and allow new root cuts
rerun (string): rerun problem if presolve infeasible or unbounded ↵

The Cplex presolve can sometimes diagnose a problem as being infeasible or unbounded. When this happens, GAMS/Cplex can, in order to get better diagnostic information, rerun the problem with presolve turned off. The GAMS solution listing will then mark variables and equations as infeasible or unbounded according to the final solution returned by the simplex algorithm. The IIS option can be used to get even more diagnostic information. The rerun option controls this behavior. Valid values are auto, yes, no and nono. The value of auto is equivalent to no if names are successfully loaded into Cplex and option IIS is set to no. In that case the Cplex messages from presolve help identify the cause of infeasibility or unboundedness in terms of GAMS variable and equation names. If names are not successfully loaded, rerun defaults to yes. Loading of GAMS names into Cplex is controlled by option Names. The value of nono only affects MIP models for which Cplex finds a feasible solution in the branch-and-bound tree but the fixed problem turns out to be infeasible. In this case the value nono also disables the rerun without presolve, while the value of no still tries this run. Feasible integer solution but an infeasible fixed problem happens in few cases and mostly with badly scaled models. If you experience this try more aggressive scaling (ScaInd) or tightening the integer feasibility tolerance EPInt. If the fixed model is infeasible only the primal solution is returned to GAMS. You can recognize this inside GAMS by checking the marginal of the objective defining constraint which is always nonzero.

Default: nono

value	meaning
auto	Automatic
yes	Rerun infeasible models with presolve turned off
no	Do not rerun infeasible models
nono	Do not rerun infeasible fixed MIP models
rhsrng (string): do right-hand-side ranging ↵

Calculate sensitivity ranges for the specified GAMS equations. Unlike most options, RHSRng can be repeated multiple times in the options file. Sensitivity range information will be produced for each GAMS equation named. Specifying all will cause range information to be produced for all equations. Range information will be printed to the beginning of the solution listing in the GAMS listing file unless option RngRestart is specified.

Default: no right-hand-side ranging is done

rinsheur (integer): relaxation induced neighborhood search frequency ↵

Cplex implements a heuristic known a Relaxation Induced Neighborhood Search (RINS) for MIP and MIQCP problems. RINS explores a neighborhood of the current incumbent to try to find a new, improved incumbent. It formulates the neighborhood exploration as a MIP, a subproblem known as the subMIP, and truncates the subMIP solution by limiting the number of nodes explored in the search tree.

Parameter RINSHeur controls how often RINS is invoked. A value of 100, for example, means that RINS is invoked every hundredth node in the tree.

Default: 0

value	meaning
-1	Disable RINS
0	Automatic
>0	Call RINS at the requested node interval
rltcuts (integer): Reformulation Linearization Technique (RLT) cuts ↵

This parameter controls the addition of cuts based on the Reformulation Linearization Technique (RLT) for nonconvex quadratic programs (QP) or mixed integer quadratic programs (MIQP) solved to global optimality. That is, the OptimalityTarget parameter must be set to 3. The RLTCuts option is not controlled by the option Cuts.

Default: 0

value	meaning
-1	Do not generate RLT cuts
0	Determined automatically
1	Generate RLT cuts moderately
2	Generate RLT cuts aggressively
3	Generate RLT cuts very aggressively
rngrestart (string): write GAMS readable ranging information file ↵

Write ranging information, in GAMS readable format, to the file named. If the file extension is GDX, the ranging information is exported as GDX file. Options ObjRng and RHSRng are used to specify which GAMS variables or equations are included.

Default: ranging information is printed to the listing file

scaind (integer): matrix scaling on/off ↵

This option influences the scaling of the problem matrix.

Default: 0

value	meaning
-1	No scaling
0	Standard scaling
An equilibration scaling method is implemented which is generally very effective.
1	Modified, more aggressive scaling method
This method can produce improvements on some problems. This scaling should be used if the problem is observed to have difficulty staying feasible during the solution process.
siftalg (integer): sifting subproblem algorithm ↵

Sets the algorithm to be used for solving sifting subproblems.

Default: 0

value	meaning
0	Automatic
1	Primal simplex
2	Dual simplex
3	Network simplex
4	Barrier
siftdisplay (integer): sifting display level ↵

Determines the amount of sifting progress information to be displayed.

Default: 1

value	meaning
0	No display
1	Display major iterations
2	Display LP subproblem information
sifting (boolean): switch for sifting from simplex optimization ↵

Default: 1

siftitlim (integer): limit on sifting iterations ↵

Sets the maximum number of sifting iterations that may be performed if convergence to optimality has not been reached.

Default: large

simdisplay (integer): simplex display level ↵

This option controls what Cplex reports (normally to the screen) during optimization. The amount of information displayed increases as the setting value increases.

Default: 1

value	meaning
0	No iteration messages are issued until the optimal solution is reported
1	An iteration log message will be issued after each refactorization
Each entry will contain the iteration count and scaled infeasibility or objective value.
2	An iteration log message will be issued after each iteration
The variables, slacks and artificials entering and leaving the basis will also be reported.
singlim (integer): limit on singularity repairs ↵

The singularity limit setting restricts the number of times Cplex will attempt to repair the basis when singularities are encountered. Once the limit is exceeded, Cplex replaces the current basis with the best factorizable basis that has been found. Any non-negative integer value is valid.

Default: 10

solnpool (string): solution pool file name ↵

The solution pool enables you to generate and store multiple solutions to a MIP problem. The option expects a GDX filename. This GDX file name contains the information about the different solutions generated by Cplex. Inside your GAMS program you can process the GDX file and read the different solution point files. Please check the GAMS/Cplex solver guide document and the example model solnpool.gms from the GAMS model library.

solnpoolagap (real): absolute tolerance for the solutions in the solution pool ↵

Sets an absolute tolerance on the objective bound for the solutions in the solution pool. Solutions that are worse (either greater in the case of a minimization, or less in the case of a maximization) than the objective of the incumbent solution according to this measure are not kept in the solution pool.

Values of the solution pool absolute gap and the solution pool relative gap SolnPoolGap may differ: For example, you may specify that solutions must be within 15 units by means of the solution pool absolute gap and also within 1% of the incumbent by means of the solution pool relative gap. A solution is accepted in the pool only if it is valid for both the relative and the absolute gaps.

The solution pool absolute gap parameter can also be used as a stopping criterion for the populate procedure: if populate cannot enumerate any more solutions that satisfy this objective quality, then it will stop. In the presence of both an absolute and a relative solution pool gap parameter, populate will stop when the smaller of the two is reached.

Default: 1.0e+75

solnpoolcapacity (integer): limits of solutions kept in the solution pool ↵

Limits the number of solutions kept in the solution pool. At most, SolnPoolCapacity solutions will be stored in the pool. Superfluous solutions are managed according to the replacement strategy set by the solution pool replacement parameter SolnPoolReplace.

The optimization (whether by MIP optimization or the populate procedure) will not stop if more than SolnPoolCapacity are generated. Instead, stopping criteria are regular node and time limits and PopulateLim, SolnPoolGap and SolnPoolAGap.

Default: 2100000000

solnpoolgap (real): relative tolerance for the solutions in the solution pool ↵

Sets a relative tolerance on the objective bound for the solutions in the solution pool. Solutions that are worse (either greater in the case of a minimization, or less in the case of a maximization) than the incumbent solution by this measure are not kept in the solution pool.

Values of the solution pool absolute gap SolnPoolAGap and the solution pool relative gap may differ: For example, you may specify that solutions must be within 15 units by means of the solution pool absolute gap and within 1% of the incumbent by means of the solution pool relative gap. A solution is accepted in the pool only if it is valid for both the relative and the absolute gaps.

The solution pool relative gap parameter can also be used as a stopping criterion for the populate procedure: if populate cannot enumerate any more solutions that satisfy this objective quality, then it will stop. In the presence of both an absolute and a relative solution pool gap parameter, populate will stop when the smaller of the two is reached.

Default: 1.0e+75

solnpoolintensity (integer): solution pool intensity for ability to produce multiple solutions ↵

Controls the trade-off between the number of solutions generated for the solution pool and the amount of time or memory consumed. This parameter applies both to MIP optimization and to the populate procedure.

Values from 1 to 4 invoke increasing effort to find larger numbers of solutions. Higher values are more expensive in terms of time and memory but are likely to yield more solutions.

Default: 0

value	meaning
0	Automatic
Its default value, 0 , lets Cplex choose which intensity to apply.
1	Mild: generate few solutions quickly
For value 1, the performance of MIP optimization is not affected. There is no slowdown and no additional consumption of memory due to this setting. However, populate will quickly generate only a small number of solutions. Generating more than a few solutions with this setting will be slow. When you are looking for a larger number of solutions, use a higher value of this parameter.
2	Moderate: generate a larger number of solutions
For value 2, some information is stored in the branch and cut tree so that it is easier to generate a larger number of solutions. This storage has an impact on memory used but does not lead to a slowdown in the performance of MIP optimization. With this value, calling populate is likely to yield a number of solutions large enough for most purposes. This value is a good choice for most models.
3	Aggressive: generate many solutions and expect performance penalty
For value 3, the algorithm is more aggressive in computing and storing information in order to generate a large number of solutions. Compared to values 1 and 2, this value will generate a larger number of solutions, but it will slow MIP optimization and increase memory consumption. Use this value only if setting this parameter to 2 does not generate enough solutions.
4	Very aggressive: enumerate all practical solutions
For value 4, the algorithm generates all solutions to your model. Even for small models, the number of possible solutions is likely to be huge; thus enumerating all of them will take time and consume a large quantity of memory.
solnpoolmerge (string): solution pool file name for merged solutions ↵

Similar to solnpool this option enables you to generate and store multiple solutions to a MIP problem. The option expects a GDX filename. This GDX file contains all variables with an additional first index (determined through SolnPoolPrefix) as parameters (Cplex only reports the primal solution). Inside your GAMS program you can process the GDX file and read all solutions in one read operation. Please check the GAMS/Cplex solver guide document for further solution pool options and the example model solmpool.gms from the GAMS model library.

solnpoolnumsym (integer): maximum number of variable symbols when writing merged solutions ↵

Default: 10

solnpoolpop (integer): methods to populate the solution pool ↵

Regular MIP optimization automatically adds incumbents to the solution pool as they are discovered. Cplex also provides a procedure known as populate specifically to generate multiple solutions. You can invoke this procedure either as an alternative to the usual MIP optimizer or as a successor to the MIP optimizer. You can also invoke this procedure many times in a row in order to explore the solution space differently (see option SolnPoolPopRepeat). In particular, you may invoke this procedure multiple times to find additional solutions, especially if the first solutions found are not satisfactory.

Default: 1

value	meaning
1	Just collect the incumbents found during regular optimization
2	Calls the populate procedure
solnpoolpopdel (string): file with solution numbers to delete from the solution pool ↵

After the GAMS program specified in SolnPoolPopRepeat determined to continue the search for alternative solutions, the file specified by this option is read in. The solution numbers present in this file will be delete from the solution pool before the populate routine is called again. The file is automatically deleted by the GAMS/Cplex link after processing.

solnpoolpoprepeat (string): method to decide if populating the solution should be repeated ↵

After the termination of the populate procedure (see option SolnPoolPop). The GAMS program specified in this option will be called which can examine the solutions in the solution pool and can decide to run the populate procedure again. If the GAMS program terminates normally (not compilation or execution time error) the search for new alternative solutions will be repeated.

solnpoolprefix (string): file name prefix for GDX solution files ↵

Default: soln

solnpoolreplace (integer): strategy for replacing a solution in the solution pool ↵

Default: 0

value	meaning
0	Replace the first solution (oldest) by the most recent solution; first in, first out
1	Replace the solution which has the worst objective
2	Replace solutions in order to build a set of diverse solutions
solutiontype (integer): type of solution (basic or non basic) for an LP or QP ↵

Specifies the type of solution (basic or non basic) that CPLEX attempts to compute for a linear program (LP) or for a quadratic program (QP). In this context, basic means having to do with the basis, and non basic applies to the variables and constraints not participating in the basis.

By default (that is, when the value of this parameter is 0 (zero) automatic), CPLEX seeks a basic solution (that is, a solution with a basis) for all linear programs (LP) and for all quadratic programs (QP).

When the value of this parameter is 1 (one), CPLEX seeks a basic solution, that is, a solution that includes a basis with a basic status for variables and constraints. In other words, CPLEX behaves the same way for the values 0 (zero) and 1 (one) of this parameter.

When the value of this parameter is 2, CPLEX seeks a pair of primal-dual solution vectors. This setting does not prevent CPLEX from producing status information, but in seeking a pair of primal-dual solution vectors, CPLEX possibly may not produce basic status information; that is, it is possible that CPLEX does not produce status information about which variables and constraints participate in the basis at this setting.

Do not use the deprecated value -1 (minus one) of the parameter barrier crossover algorithm to turn off crossover of the barrier algorithm but use this parameter to indicate that a primal-dual pair is sufficient.

Default: 0

value	meaning
0	Automatic
1	Basic solution
2	primal-dual pair
solvefinal (boolean): switch to solve the problem with fixed discrete variables ↵

Sometimes the solution process after the branch-and-cut that solves the problem with fixed discrete variables takes a long time and the user is interested in the primal values of the solution only. In these cases, solvefinal can be used to turn this final solve off. Without the final solve no proper marginal values are available and only NAs are returned to GAMS.

Default: 1

value	meaning
0	Do not solve the fixed problem
1	Solve the fixed problem and return duals
sos1reform (integer): automatic logarithmic reformulation of special ordered sets of type 1 (SOS1) ↵

Default: 0

sos2reform (integer): automatic logarithmic reformulation of special ordered sets of type 2 (SOS2) ↵

Default: 0

startalg (integer): MIP starting algorithm ↵

Selects the algorithm to use for the initial relaxation of a MIP.

Default: 0

value	meaning
0	Automatic
1	Primal simplex
2	Dual simplex
3	Network simplex
4	Barrier
5	Sifting
6	Concurrent
strongcandlim (integer): size of the candidates list for strong branching ↵

Limit on the length of the candidate list for strong branching (VarSel = 3).

Default: 10

strongitlim (integer): limit on iterations per branch for strong branching ↵

Limit on the number of iterations per branch in strong branching (VarSel = 3). The default value of 0 causes the limit to be chosen automatically which is normally satisfactory. Try reducing this value if the time per node seems excessive. Try increasing this value if the time per node is reasonable but Cplex is making little progress.

Default: 0

subalg (integer): algorithm for subproblems ↵

Strategy for solving linear sub-problems at each node.

Default: 0

value	meaning
0	Automatic
1	Primal simplex
2	Dual simplex
3	Network optimizer followed by dual simplex
4	Barrier with crossover
5	Sifting
submipnodelim (integer): limit on number of nodes in an RINS subMIP ↵

Controls the number of nodes explored in an RINS subMIP. See option RINSHeur.

Default: 500

submipscale (integer): scale the problem matrix when CPLEX solves a subMIP during MIP optimization ↵

Default: 0

value	meaning
-1	No scaling
0	Standard scaling
1	Modified, more aggressive scaling method
submipstartalg (integer): starting algorithm for a subMIP of a MIP ↵

Default: 0

value	meaning
0	Automatic
1	Primal simplex
2	Dual simplex
3	Network simplex
4	Barrier
5	Sifting
submipsubalg (integer): algorithm for subproblems of a subMIP of a MIP ↵

Default: 0

value	meaning
0	Automatic
1	Primal simplex
2	Dual simplex
3	Network optimizer followed by dual simplex
4	Barrier with crossover
5	Sifting
symmetry (integer): symmetry breaking cuts ↵

Determines whether symmetry breaking cuts may be added, during the preprocessing phase, to a MIP model.

Default: -1

value	meaning
-1	Automatic
0	Turn off symmetry breaking
1	Moderate level of symmetry breaking
2	Aggressive level of symmetry breaking
3	Very aggressive level of symmetry breaking
4	Highly aggressive level of symmetry breaking
5	Extremely aggressive level of symmetry breaking
threads (integer): global default thread count ↵

Synonym: gthreads

Default number of parallel threads allowed for any solution method. Negative values are interpreted as the number of cores to leave free so setting threads to -1 leaves one core free for other tasks. Cplex does not understand negative values for the threads parameter. GAMS/Cplex will translate this into a positive number by applying the following formula: max(1,number of cores-|threads|). Setting threads to 0 lets Cplex use at most 32 threads or the number of cores of the machine, whichever is smaller.

Default: GAMS Threads

tilim (real): overrides the GAMS ResLim option ↵

Synonym: reslim

The time limit setting determines the amount of time in seconds that Cplex will continue to solve a problem. This Cplex option overrides the GAMS ResLim option. Any non-negative value is valid.

Default: GAMS ResLim

trelim (real): maximum space in memory for tree ↵

Sets an absolute upper limit on the size (in megabytes) of the branch and cut tree. If this limit is exceeded, Cplex terminates optimization.

Default: 1.0e+75

tuning (string): invokes parameter tuning tool ↵

Invokes the Cplex parameter tuning tool. The mandatory value following the keyword specifies a GAMS/Cplex option file. All options found in this option file will be used but not modified during the tuning. A sequence of file names specifying existing problem files may follow the option file name. The files can be in LP, MPS or SAV format. Cplex will tune the parameters either for the problem provided by GAMS (no additional problem files specified) or for the suite of problems listed after the GAMS/Cplex option file name without considering the problem provided by GAMS. Due to technical reasons a single option input line is limited by 256 characters. If the list of model files exceeds this length you can provide a second, third, ... line starting again with keyword tuning and a list of model instance files.

The result of such a tuning run is the updated GAMS/Cplex option file with a tuned set of parameters. The solver and model status returned to GAMS will be NORMAL COMPLETION and NO SOLUTION. More details on Cplex tuning can be found on IBM's web page. Tuning is incompatible with the BCH facility and other advanced features of GAMS/Cplex.

tuningdettilim (real): tuning deterministic time limit per model or suite ↵

Default: 1.0e+75

tuningdisplay (integer): level of information reported by the tuning tool ↵

Specifies the level of information reported by the tuning tool as it works.

Default: 1

value	meaning
0	Turn off display
1	Display standard minimal reporting
2	Display standard report plus parameter settings being tried
3	Display exhaustive report and log
tuningmeasure (integer): measure for evaluating progress for a suite of models ↵

Controls the measure for evaluating progress when a suite of models is being tuned. Choices are mean average and minmax of time to compare different parameter sets over a suite of models

Default: 1

value	meaning
1	mean average
2	minmax
tuningrepeat (integer): number of times tuning is to be repeated on perturbed versions ↵

Specifies the number of times tuning is to be repeated on perturbed versions of a given problem. The problem is perturbed automatically by Cplex permuting its rows and columns. This repetition is helpful when only one problem is being tuned, as repeated perturbation and re-tuning may lead to more robust tuning results. This parameter applies to only one problem in a tuning session.

Default: 1

tuningtilim (real): tuning time limit per model or suite ↵

Sets a time limit per model and per test set (that is, suite of models).

As an example, suppose that you want to spend an overall amount of time tuning the parameter settings for a given model, say, 2000 seconds. Also suppose that you want Cplex to make multiple attempts within that overall time limit to tune the parameter settings for your model. Suppose further that you want to set a time limit on each of those attempts, say, 200 seconds per attempt. In this case you need to specify an overall time limit of 2000 using GAMS option reslim or Cplex option TiLim and tuningtilim to 200.

Default: 0.2*GAMS ResLim

upperobjstop (real): in a maximization MILP or MIQP, the solver will abort the optimization process as soon as it finds a solution of value greater than or equal to the specified value ↵

Default: 1e75

usercallparmfile (string): Command-line parameter include file used in GAMS command-line calls triggered by BCH ↵

.usercut (boolean): User cut activation ↵

Determines whether a linear constraint is treated as a user cut. At the beginning of the MIP solution process, any constraint whose usercut attribute is set to 1 (the default value is 0) is removed from the model and placed in the user cut pool. User cuts may be used by CPLEX at any time to improve the solution process. There is no guarantee that they are actually used.

The user cut pool is only active if option usercutpool is enabled and are specified through the option .usercut. The syntax for dot options is explained in the Introduction chapter of the Solver Manual.

Default: 0

usercutcall (string): the GAMS command line to call the cut generator ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

usercutfirst (integer): calls the cut generator for the first n nodes ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 10

usercutfreq (integer): determines the frequency of the cut generator model calls ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 10

usercutinterval (integer): determines the interval when to apply the multiplier for the frequency of the cut generator model calls ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 100

usercutmult (integer): determines the multiplier for the frequency of the cut generator model calls ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 2

usercutnewint (boolean): calls the cut generator if the solver found a new integer feasible solution ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 1

usercutpool (boolean): Indicator to use user cuts ↵

Default: 0

usergdxin (string): the name of the GDX file read back into Cplex ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: bchin.gdx

usergdxname (string): the name of the GDX file exported from the solver with the solution at the node ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: bchout.gdx

usergdxnameinc (string): the name of the GDX file exported from the solver with the incumbent solution ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: bchout_i.gdx

usergdxprefix (string): prefixes usergdxin, usergdxname, and usergdxnameinc ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

usergdxsol (string): the name of the GDX file exported by Cplex to store the solution of extra columns ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: bchsol.gdx

userheurcall (string): the GAMS command line to call the heuristic ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

userheurfirst (integer): calls the heuristic for the first n nodes ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 10

userheurfreq (integer): determines the frequency of the heuristic model calls ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 10

userheurinterval (integer): determines the interval when to apply the multiplier for the frequency of the heuristic model calls ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 100

userheurmult (integer): determines the multiplier for the frequency of the heuristic model calls ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 2

userheurnewint (boolean): calls the heuristic if the solver found a new integer feasible solution ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 1

userheurobjfirst (integer): Similar to UserHeurFirst but only calls the heuristic if the relaxed objective promises an improvement ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 0

userincbcall (string): the GAMS command line to call the incumbent checking program ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

userincbicall (string): the GAMS command line to call the incumbent reporting program ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

userjobid (string): postfixes lf, o on call adds –userjobid to the call. Postfixes gdxname, gdxnameinc and gdxin ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

userkeep (boolean): calls gamskeep instead of gams ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Default: 0

userlazyconcall (string): the GAMS command line to call the lazy constraint generator ↵

More info is available in chapter The GAMS Branch-and-Cut-and-Heuristic Facility.

Note: There is no guarantee that CPLEX will use all of the added violated lazy constraints provided due to technical and/or efficiency reasons. It may thus happen that a later candidate solution violates previously provided lazy constraints. In this case consider passing the constraint again.

varsel (integer): variable selection strategy at each node ↵

This option is used to set the rule for selecting the branching variable at the node which has been selected for branching. The default value of 0 allows Cplex to select the best rule based on the problem and its progress.

Default: 0

value	meaning
-1	Branch on variable with minimum infeasibility
This rule may lead more quickly to a first integer feasible solution, but will usually be slower overall to reach the optimal integer solution.
0	Branch variable automatically selected
1	Branch on variable with maximum infeasibility
This rule forces larger changes earlier in the tree, which tends to produce faster overall times to reach the optimal integer solution.
2	Branch based on pseudo costs
Generally, the pseudo-cost setting is more effective when the problem contains complex trade-offs and the dual values have an economic interpretation.
3	Strong Branching
This setting causes variable selection based on partially solving a number of subproblems with tentative branches to see which branch is most promising. This is often effective on large, difficult problems.
4	Branch based on pseudo reduced costs
warninglimit (integer): determines how many times warnings of a specific type (datacheck=2) will be displayed ↵

By default, when modeling assistance is turned on via the data consistency checking parameter, CPLEX will display 10 warnings for a given modeling issue and then omit the rest. This parameter controls this limit and allows the user to display all of the warnings if desired. In order to see all warnings change the value to its negative.

Default: 10

workdir (string): directory for working files ↵

The name of an existing directory into which Cplex may store temporary working files. Used for MIP node files and by out-of-core Barrier.

Default: current or project directory

workeralgorithm (integer): set method for optimizing benders subproblems ↵

Default: 0

value	meaning
0	Automatic
1	Primal Simplex
2	Dual Simplex
3	Network Simplex
4	Barrier
5	Sifting
workmem (real): memory available for working storage ↵

Upper limit on the amount of memory, in megabytes, that Cplex is permitted to use for working files. See parameter WorkDir.

Default: 2048.0

writeannotation (string): produce a Cplex annotation file ↵

writebas (string): produce a Cplex basis file ↵

Write a basis file.

writeflt (string): produce a Cplex solution pool filter file ↵

Write the diversity filter to a Cplex FLT file.

writelp (string): produce a Cplex LP file ↵

Write a file in Cplex LP format.

writemps (string): produce a Cplex MPS file ↵

Write an MPS problem file.

writemst (string): produce a Cplex mst file ↵

Write a Cplex MST (containing the MIP start) file.

writeord (string): produce a Cplex ord file ↵

Write a Cplex ORD (containing priority and branch direction information) file.

writeparam (string): produce a Cplex parameter file with all active options ↵

Write a Cplex parameter (containing all modified Cplex options) file.

writepre (string): produce a Cplex LP/MPS/SAV file of the presolved problem ↵

Synonym: writepremps

Write a Cplex LP, MPS, or SAV file of the presolved problem. The file extension determines the problem format. For example, WritePre presolved.lp creates a file presolved.lp in Cplex LP format.

writeprob (string): produce a Cplex problem file and inferrs the type from the extension ↵

Write a problem file in a format inferred from the extension. Possible formats are

SAV: Binary matrix and basis file
MPS: MPS format
LP: CPLEX LP format with names modified to conform to LP format
REW: MPS format, with all names changed to generic names
RLP: LP format, with all names changed to generic names
ALP: LP format, with generic name of each variable, type of each variable, bound of each variable If the file name ends with .bz2 or .gz, a compressed file is written.
writesav (string): produce a Cplex binary problem file ↵

Write a binary problem file.

zerohalfcuts (integer): zero-half cuts ↵

Decides whether or not to generate zero-half cuts for the problem. The value 0, the default, specifies that the attempt to generate zero-half cuts should continue only if it seems to be helping. If the dual bound of your model does not make sufficient progress, consider setting this parameter to 2 to generate zero-half cuts more aggressively.

Default: 0

value	meaning
-1	Off
0	Automatic
1	Generate zero-half cuts moderately
2	Generate zero-half cuts aggressively
