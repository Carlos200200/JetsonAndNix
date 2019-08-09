# NVIDIA TX1 Cluster

This entry explains the steps to perform the project of making a Cluster of NVIDIA Jetson TX1 to perform HPC Computing and also integrate them on GRID5000.

**Note**: Maybe sometimes for the time of making quick notes i'll post in spanish, but then i will modify them.

## Main Goals:

The main goals of the project will explain next:
1. Using the NVIDIA Jetson TX1 on HPC
2. Make a Cluster and use functional package manager to make it run.

## Part 1: Getting Started

For this part there are little goals to start the project.
 1. Getting acquaintance of Nvidia Jetson TX1 system
-Test the factory set.
-Update and test their capabilities
2. Getting acquaintance of Git (for project needs)

### Test the factory set of TX1

The only thing to point out is to not power on the TX1 if you haven't connect all the external cables (usb, hdmi, so on) because can harm the system.

It says is aim to perform Deep Learning programming thanks to all their CUDA cores, have a camera for recognition of objects and an ARM64 architecture, different for the TK1.

Trying to update to ubuntu 16.04 the system halts everytime from the login page, so i don't know if it was for how i perfom the update of simply the system can't be updated in that way.

Other task was to send it internet connection through a PC that has ethernet connection.

Share connection wireless to ethernet

- You can use this web site:
https://major.io/2015/03/29/share-a-wireless-connection-via-ethernet-in-gnome-3-14/

- But be careful with the dnsmasq that has a problem with the daemon that create the DNS and the mask on the other device, so you have to unistall it.

### Avril 12 2018 (L4T and CUDA)

Because the system halted i won't be able to use it, so i have to flash it again with its development kit and we start with **L4T R24.2.2**.

	Installing L4T

Software for Jetson Series
https://developer.nvidia.com/embedded/develop/software

Jetson Download Center
https://developer.nvidia.com/embedded/downloads

Current Version: R28.2

Step by Step
http://developer.download.nvidia.com/embedded/L4T/r23_Release_v1.0/l4t_quick_start_guide.txt

> That step by step works, the problem is that something is missing on TX1 that didn't work on L4T 24.2.2

(Above failed) Failed to flash TX1
https://devtalk.nvidia.com/default/topic/1015744/failed-to-flash-to-tx1/
	
(Above failed) EXT4-fs error loading journal
https://devtalk.nvidia.com/default/topic/1026155/tx1-won-t-boot-after-flushing-r24-2-2-into-it

> **Note:** It doesn't work on L4T 24.2.2 but it does on L4T R28.1 with the step by step telling above!

	Installing CUDA

We test with CUDA 7 and it doesn't work because diferents architectures (armhf vs arm64). 

The new versions require to install Jetpack because there is no direct links, but affortunately people find a method to find that links (since Jetpack only works on ubuntu 14.04 and probably on 16.04).

Metodo para extraer los links del jetpack sin instalarlo:
http://warppipe.net/blog/installing-cuda-packages-on-jetson-boards/

Pag. donde se saco el link del CUDA:
https://devtalk.nvidia.com/default/topic/982848/jetson-tx1/tx1-specific-arm64-deb-repo-for-cuda-8/post/5063154/

- Link del CUDA: wget http://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/006/linux-x64/cuda-repo-l4t-8-0-local_8.0.34-1_arm64.deb

- I didn't use the method, only used wget to perform that link and it works.

Instalar CUDA:
https://elinux.org/Jetson/Installing_CUDA

- Only make the steps of above and is going to be ok, its simple.
- I tested the CUDA Samples and they ran without complains.

### Avril 13 2018 (GIT Repository)

In this part i getting acquaintance of GIT, what is it, how it works and a basic tutorial, with the goal to have a repository of this kind in the late part of the project.

	GIT

Que es GIT: Intro
https://www.atlassian.com/git/tutorials/what-is-version-control

Git and Github | Practical Course from Scratch (espanol)
https://www.youtube.com/watch?v=HiXLkL42tMU

Branches on github (don't see all)
https://www.youtube.com/watch?v=oPpnCh7InLY


### Avril 16 2018

AVRIL 16 2018

Goals: 

 1. What's a functional language? 
 2. MPICH on linux host 	MPICH on TX1
 3. What's NIX?

First we looked for the definition of Functional Languages, looked examples and read some definitions on Quora, for example.

	Functional Language

Functional languages is another paradigm where you use *values* instead of *variables*. The values didn't change in the time, so the idea is to know what is going to do the program not how it will do.

Lambda in python is a good start to make functional statements, but Scala is another language that supports functional programming.

> In this moment i don't know what i need for the functional language, but i suspect is when we'll build package on Nix they focus on Nix Expressions that is sort of a class of Functional Statements.

	Other Things

- We read NIX documentation once again.  
- We installed NIX on Host PC and make it works with the simple example.
	 - No problem with the system
 - We tried to use mpicc (Compiles and links MPI programs written in C) within the library on Nix
	- Didn't work, missing libraries and links -> gonna ask for help.
	- Don't know with it didn't work but i found that there is a library in another repository here on Grenoble CIMENT (for HPC).
- Installed mpich (MPI) on Jetson TX1
	- Successfull -> Compile and run Hello_WorldMPI.c

Next: Install openmp for multiple processors; MPI is for message between processors, OpenMP is for cores.

### Avril 17 2018

Goals:
1. Why Jetson TX1
2. Installing NIX on Jetson TX1 

The idea is have a first look about the features that make Nvidia Jetson TX1 very accurate to HPC on clusters. Our brush look told us that is aim to IA performance like Deep Learning, Neural Networks, etc...

Looking for some papers we found some fields that are studying with Jetson TX1:
1. Deep Neural Network	
2. Real-Time Computer Vision Workloads
3. Speech Recognition
4. Proton Testing (Single-Event Effects)
5. Molecular Dynamics
6. Robotics
7. Convolutional Neural Networks

And then we found a paper called *"Understanding the Role of GPGPU-Accelerated SoCs-based ARM Cluster"*, that said they studied a cluster of NVIDIA TX1 and make performance testing.

	About the Paper

 - Told that Montblanc prototype only use CPU cores instead of the GPU cores.
 - Use and develop a benchmark called ClusterSoCBench.
 - Concluded that "Compared with the server-class ARM SoC a large number of ARM cores on a single chip not neccesarily guarantee better performance due to the poor performance of the branch predictor and L2 Cache for the Server-class ARM SoCs".
 - They focus on a cluster using a 10Gb/s NIC instead of the common 1Gb/s.
- **CUDA has 3 Memory Transfer Model**:
	1. **Host and Device Memory**: Different memories, different address space so have to change in a moment when need to recall an address for one place to another.
	2. **Zero Copy**: Enable device threads to directly access host memory (without copy temporally in a place -don't know where-).
	3. **Unified Memory**: 	Both memories worked as one, so is a directly address specification without a conversion.
- Use 16 Nvidia Jetson TX1 with a connection of 10Gb/s and test with the 1Gb/s one.
- **Stream Benchmark**: Reveal max. memory bandwith. In this case was 11,72Gb/2 on CPU Cores and 21Gb/2 on GPU Cores.
- Putting more bandwith on NIC upgrade the features of the system:
	- *Measure using the **iperf tool***: The **Average Throughtput** was from 0,53Gb/s (@1Gb/s) to 3,13Gb/s (@10Gb/s).
	- *Measure using **Latency-Bandwith Benchmark***: The **Average Latency** (*Ping Pong Latency*) was from 0,4ms (@1Gb/s) to 0,05ms (@10Gb/s).
- They said that is a **Lack of Standard Benchmark Suite**.
- They have github for the Benchmark applications.
- Use also NAS Parallel Benchmark Class C (Max for Jetson TX1).
- Installed CUDA 7, Gcc 5,4. Also OpenMpi and OpenBLAS from source.
- The Zero-Copy method put down the performance of TX1 because there is a Bypass for Cache-Coherency.
- **AlexNet** and **googlenet** benchmark.

Things to Learn:
1. What's Multiplexing?
2. What's the "Mobile-Class SoCs" that told in the paper?

Also i need the paper catalog for papers review.

	Nix on NVIDIA Jetson TX1

Fortunately, it works!

1. Installing Curl on TX1
2. Use Curl for the address of Nix
	- It works! No errors, no warnings
	- Testing Hello world, ok.
	- Testing nix-shell, ok
	- Testing nix-env -qa, ...ok

### Avril 18 2018

Goal:
1. Getting started with Nix Packages
2. Some benchmarks and test

We have to start with the packages but there is some things you should have learn first.

	NIX Packages

Create a package could be a little difficult, but you can resume it in 3 parts:
1. Nix Expression
2. Build Script
3. Arguments - Variables

> Please note that Nix is a package manager that have scripts to install software and associate the program language, and Nixpkgs is the common repository which have Emacs, C, C++, and so on.

So, before installing anything on Nix, is necessary to learn the system basics. And that is the element called NixExpressions that use nixpkgs to build a program.

	Nix Expressions

To start to practice Nix Expressions, we can use the Nix enviroment to test things, so, we use **nix-env -i nix-repl** and execute it with **nix-repl**.

Something in consideration is that **Nix is not a general purpose language**, is a domain-specific languague for writing packages.

```Nix
Operations
	> 2+1 // 3
	> 5+4 // 9
	> 2-1 // Don't work, Nix seems it like "2-1"
	> 2 - 1 // 1
	> 2/4 // Don't work, have to use a function set

Interpolate:
	> ${foo}
	> word = "hola"
	> "${foo}"  // hola, only works in that way, no more ways

Lists:
	> [exp1 exp2 ..... expn]
	> [2 "hola" true 5]     // Can put whatever expression you want
	> [ (2+3) ]   // [5] ; it resolve the expresion if is necessary

Attribute Sets:
	> s = {word="hola mundo"; "123"="como esta"} // s, word, "123" is an identificator
	> 123 = "hola"  // don't work because 123 is not an identificator (is a number)
	> {a=20; b=50; c=a+b} // Don't work for the recursivity
	> rec {a=20; b=50; c=a+b}  // Recursive attr set, now it works

if expressions:
	> a=2
	> b=6
	> if a>b then "a es mayor" else "a es menor" // "a es menor"

Let expressions:
	> let a=b; b=2; in a*b  // 20, notice the ";"
	> let a=5; in let a=20; in a // 20

With expression:
	> Lista={a=10;b=20}
	> Lista.a * Lista.b  // 200
	> with Lista; a*b  // 200, use the expr to simplify the big expression

Laziness:
	> let a=builtins.div 4 0 ; b=6; in b  // Only evaluate when is needed

Functions:
	> <parameter>: <expression>
	> x:x*2  // lambda, is an expression
	> fun = x:x*2  // Identificator to the expre
	> fun 5        // 10, evaluate the function
	> fun = x: y: x*y // Nested functions
	> fun 5 4         // 20, it works with parenthesis too
```
> Be carefull, all above is  an expression, not statement:
> 1. Expression: Inmutable, dont' change, only 1 result
> 2. Statement: Mutable, something that could change

### Avril 19 2018

Goals:
1. Nix Pills: Continue
2. Benchmarking: Introduction

All these information are in a web page called Nix Pills at https://nixos.org/nixos/nix-pills/functions-and-imports.html

```Nix
Arguments Set:
	> mul = s: s.a *s.b
	> mul = {a=3; b=4} // Send the 2 parameters that ask in a set.
	
	> mul = {a,b}: a*b // Explicit parameter call
	> mul = {a=3; b=4}

Default and Variatic Attributes
	> mul = {a,b ? 2}: a*b  // Specifies a default value
	> mul {a=5}  // 10
	
	> mul = {a,b, ...}: a*b // specifies more atributes that the ones they wait.
	> mul {a=5; b=10; c=20}  // 50

	> s@{a,b, ...}: a*b*s.c // To operate with c (in other cases you can access other parameters).
```

	Imports on Nix

The import function is a build-in and provides a way to parse a *.Nix File*. Normally you define the components on *.Nix File* and then compose those files.

Supose we have the following files in our directory with the contents that present below:

```Nix
a.nix		b.nix		mul.nix     // Files
  3		      4 		a:b:a*b     // Content
```
Now we can import the libraries and execute the function on Nix with the Nix Expressions:

```Nix
	> a = import ./a.nix	// import and put the expr on a
	> b = import ./b.nix	// the above but with b
	> mul = import ./mul.nix  // now the mul.nix
	> mul a b    			// finally a Nix expression
```
If you want to pass values (as in mul {a=5, b=10}), you have to create a function in the file (let x in ... don't work).
```Nix
Test.nix
	{a, b ? 3, trueMsg ? "yes", falseMsg ? "no"}:
	if a>b
		then builtins.trace trueMsg true
		else builtins.trace falseMsg false
```
```Nix
	> import ./Text.nix {a=5; trueMsg="Es mayor";}
	  ----Function----  ------Parameters----------
	  // trace: es mayor, true
```

	Benchmarks

The benchmarks are programs that evaluate the system performance, in this case the HPC Systems.

In this part we only list some benchmarks. We don't know if we go to use all of them but is a first approach to the HPC Benchmarks to use on Jetson TX1.

1. Stream (Memory Bandwidth):
	- Memory Bandwidth
	- Giving in Mb/sec
	
2. Linpack (HPL):
	- Consist on solving 1000x1000 system of simultaneous equations.
	- Can be written in Fortran or C
	- More equations: 2500x2500 for scalability

3. Random Access:
	- Measures the rate of integer random updates of memory.
	- GUPS (Giga-updates per second)

4. FFT:
	- Measures the floating-point rate of execution of double precision complex one-dimensional discrete fourier transform (DFT).

5. Effective Bandwidth Benchmark
	- Measure latency and bandwidth of number of simultaneous communication patterns.
	- The file is called b_eff

### Avril 20 2018

Embedded systems benchmarks:

- Coffee Alex Not: Deep learning Benchmark
- Phoronix Test Suite: Another benchmark


### Avril 23 2018

	Nix Pills Chapter 6: Our 1st Derivation

Derivation built-in function:
- Name: Hash name
- System: System where can be built. Ej: "x86_64-linux".
- Builder: Binary program that build the derivation

```Nix
	> builtins.currentSystem // Tell the system as  "x86_64-linux"
```

Some allegories of Nix files:
- .Nix -> .c files
- .drv -> .0 files
- out paths -> product of the build

```Nix
	> d = derivation {name=...., ....}
	> :b d  // Method 1 to derive a package
	
	> nix-store -r /nix/store/ .... -myname.drv  // Method 2 to derive a package
```

	Our First Derivation

Supose a file on /home (or where you want):

```
builder.sh
	declare -xp
	echo foo > $out
```
```Nix
	> :l <nixpkgs>
	> "${bash}"    // Give an drv Nix on bash
	> d = derivation {name="foo"; builder= "${bash}/bin/bash"; args=[./builder.sh]; system= builtins.currentSystem;}
	> :b d // Do the derivation
```

	Packaging a simple C program

```C
simple.c
	void main()
	{
		puts("simple");
	}
```
```Nix
simple_builder.sh
	export PATH = "$coreutils/bin:$gcc/bin"
	mkdir $out
	gcc -o $ out/simple $src
```
```
	> :l <nixpkgs>
	> simple = derivation{... args=[./simple_builder.sh]; gcc=gcc; src=./simple.c;}
```
At the end you can execute in the terminal like /nix....-simple/simple.

> Lo mismo sucede con el builder afuera de relv y con nix-build. 


### Avril 24 2018

	Olivier Richard Meeting (Part 2)
	
	1. GitLab/Inria
	2. Using CAS for the papers
	3. We have to be sure about little benchmarks than have a lot of them and not be sure about it.
	4. From the paper "Understanding the Role of GPGPU-Accelerated SoCs-based ARM Cluster" we can replicate and validate the experiments.

	* Learn (read): Easy Build, Spack, GUIX
	* Document to structure my stuff.

https://gitlab.inria.fr 
https://gitlab.inria.fr/cirkus/internships/carlos

Proxy: https://istpac.inria.fr/pac/roc.pac

olivier.richard@imag.fr

###  Avril 25 2018

WTF is a container?
https://beta.techcrunch.com/2016/10/16/wtf-is-a-container/?_ga=2.146515444.1727589572.1524669551-1173643629.1524669551

Sylabs launches Singularity Pro, a container platform for high-performance computing
https://techcrunch.com/2018/02/08/sylabs-launches-singularity-pro-a-container-platform-for-high-performance-computing/

-Revisar lo de SINGULARITY

To Do: 
-	ISC travel application
-	Marksman wiki
-	Poster


### Mai 2 2018

Goals:
- Easy Build
- Spack
- GUIX

GitLab Inria Repository

> Git add < file >
> Git commit -m "< mensaje de lo que hizo >"
> Git push -u origin master

	Easy Build

- Software build and installation framework that allows to manage scientific software on HPC systems in an efficient way.
- Flexible framework, automates software build, builds recipes human readable.
- Easily configurable, build software in parallel.

Paper Workshop: Modern Scientific Software Management using EasyBuild and Lmod.

Toolchains:
- Compiler toolchains that handle build and installation processes
	- Like GCC, Intel, Clang, CUDA
	- MPI, MPICH, OpenMPI
	- OpenBLAS, ScalaPACK, FFTW
- Dummy toolchain = empty toolchain.

_Typical Workflow Example_

The example uses a WRF (weather research and forecasting) that has plenty software dependencies.

1. Search which easyconfigs are available for the program.
```
	$ eb -S WRF
```
2. Get an overview of the planned installations of the selected easyconfig file
```
	$ eb WRF-3.5.1-goolf-1.4.10-dmpar.eb -Dr
```
3. Build and installing in 1 command line
```
	$ eb WRF-3.5.1-goolf-1.4.10-dmpar.eb --robot
```
Once the installation is succed, modules will be available for the program and all of its dependencies.

*Installing*

1. Download EasyBuild boopstrap scrip
2. Bootstrap EasyBuild into $Home/EasyBuild
3. Extend $ModulePATH and load EasyBuild module.

More detailed steps in the file or journal.

	Videos to make a paper

Videos of making a paper for the other class (and also help for my english writing). This is a resume, so main ideas are in the journal notebook.

*Video: 10 tips for Writing a truly terrible Journal Article*

1. Refuse to read the previous literature published in your field
2. Take the lazy route and plagiarize
3. Omit key article components
4. Disrespect previous publications
5. Overestimate your contribution
6. Excel in ambiguity and inconsistency
7. Apply incorrect referencing of statements
8. Proper subjective over Objective Statements
9. Give little care to grammar, spelling, figures and tables.
10. Ignore editor and reviewer comments.

*10 Additional tips: What you should do*

1. Carefully select the **most appropriate journal**
2.  First decide **where** you want to publish, then write your paper based on **Journal Guidelines**
3.  Follow the rule: "One paper, one message"
4.  Select an **attractive and descriptive title**. Most scientifics saw the title only.
5.  Figures are seductive items. Should be as attractive and clear as possible. Readers saw figures and convice to read/cite the paper.
6.  Be honest and modest. (Research difficulties > success)
7.  Start with structure with items/bullets (title, state of art, etc...) rather than sentence + sentence.
8.  Become a reviewer
9.  Be polite and respectful
10. Also cite your own work, when relevant, or future publications.

*"It's not the honors and the prizes and the fancy outsides of life which ultimately nourish our souls. It's the knowing that we can be trusted, that we never have to fear the truth, that the bedrock of our very being is good stuff"* - Rogers 2001


### Mai 3 2018

Goal:
- Paper search (where to put my paper)
- Easybuild and Spack

Continuacion de Easy build y ahora revision de spack para el mini-cluster

	EasyBuild (part 2)

- Controls how to build a program (dependencies, etc...)
- Co-existence of versions/builds via dedicated installation prefix and module files.
- Building software in parallel
- **Be Careful!** : It's a framework
- Target Audience: HPC user support system.

Works in some moments but have some problems with lmod, having to reinstalling it once i start again (putting the PC off)

	Spack

-  It's a software configuration manager (like NIX)
- HPC focus (distribuited systems, supercomputers)
- MPI requires an external link (not building in spack)
- BLAS/LAPLACK requires and external link of building.

Next step is to test EasyBuild and Spack on NVIDIA Jetson TX1 and see if it works well.

	Elements of Style for Writing

- Always put the *Topical Position* before the *Stress Position*, that is, the old info before the new one.
- Put statements in a positive form
	- He usually come late (is good)
	- He is not very often on time (not good)

Typical errors:
- "This" unqualified. It doesn't matter if its obvious is better for the reader to read without guessing.
```
"We form this tube...."
```
- Too many propositional phrases
```
"Model simulation of the ocean...." (X)
"An Ocean Model simulation..." (y)
```
- Subjective or Judgmental Adjectives
```
"We use a simple model ...." //What is simple?
"We use a idealized model..."// Ah, it's not "Real"
```
- Be careful expressing thoughts or emotions.
```
"We believe this model..." // Believe?
"We shown through our analysis..." // More scientific
```

### Mai 15 2018

Goal:
- Lecture of papers about ARM architecture and performance
- Understand some concepts of the architecture/test of ARM-based devices.

Resume (important Info) of the Papers

	Tibidabo: Make The Case for an ARM-Based HPC System

- It said that it's the 1st large-scale cluster with ARM processors.
- Conclusion: HPC Application scale well without tuning
- Start saying the problem of energy cost than few years of work could cost more than the infrastructure acquisition.
- Be careful: GPU devices are not mean of its CPU to use in bigger problems -> They have lack of vector floating point-unit and NOT tuning for HPC.
	- "Putting ARM systems doesn't guaranteed the CPU use completely because it's not designed for that".
- So, why use? -> because the ARM processors and for their energy. Data centers and Cloud Computing environment are constrained by I/O and memory subsystems, NOT CPU performance.
- Bell's Law: New devices use components of low cost but with high performance with the same price.
- Tibidabo is a prototype, don't want to compete with High-Class HPC centers.
- The mayor problem when build HPC Systems form Low-power parts is _**that the system integration glue takes more power than the microprocessor cores themselves**_.
- Use Q7 modules. The Q7 modules are in Q7 nodes. 8 nodes is a blade. They put 16 blades (128 nodes) in a Rack. All are connected with a tree topology.
- Turn off all the things hat no need including hard disk.
- Definitions:
	- Weak Scalability: You need more nodes to achieve the same time of a problem that have more data. (problem limited to available memory).
	- Strong Scalability: The time of a fixed problem is reduced when you use more nodes.
- They test with HPL, changing frequencies to see changes.
- Conclusion: More bandwidth benefit cortexA15 but no has negligible improvement on CortexA9
- So, for improvement they need more cores, chips with multicore.
- Better energy efficiency with SIMD units.
- Have balance between memory bandwidth ratio. New package-package memory to access memory better.
- Putting more than 1Gb/s bandwidth in A15 is good, because is a bottleneck.

They said its the 1st large-scale cluster with ARM processors and in other papers said that they didn't focus on GPU instead only CPU working (but with some reason).

	The Spack Package Manager: Bringing Order to HPC Software Chaos

Spack: Allows any number of build to coexist on same system, ensures that installed packages can find their dependencies, regardless of the environment.

```
MetaBuild Systems: Contractors, WAF, MiXDown
	Good: Package MAnager, single board
	BAd: LArge package repository, combinational Versioning
```
```
Traditional PAckage MAnager: RPM, YUM,APT, Anaconda
	Good: Package manager automate the installation complex set of software programm.
	BAd: SIngle, inflexible location, root privileges
Port SYstems: GEntoo, BSD Ports, MAcPorts, Homebrew
	Good: BUild pkg from source insteas pre-build binaries
	BAd: Burden of pakgs to create conflict
```
```
Virtual MAchines and Contai8ners: VM, Linux Containers
	Good: EAch user persionalized their environ,ment
	BAd: 1VM each configuration
```
```
ENvironment Modules and RPATH's: Lmod, RPATH
	Good: Provides software hieraquies, allow users load a pakg if know requieres
```
```
MOdern PAckage MAngaers: Smithy, Nix, EAsyBUild, HAshDIst
	Good: Multi-configuration builds
	BAd: NOt human readable pakgs.
```

The main limitations are the **composability**, that is, how to deal with inter-relationships between components (versions of packages).

	GPU Clusters for HPC

- There are 3 principal components used in GPU Cluster:
	- Host Nodes
	- GPUs
	- Interconnect
- They expect that GPU carry more part of the calculation.
	- Also, host memory, PCI Bus, Network performance hast to matched with GPU performance for well-balance system.
- The key requirement to acceleration from GPU subroutine libraries is minimize I/O between host and GPU.
- Cuda C, OPenCL, PGI x86, GPU Fortran, C99 Compiler
- MPI-Cuda C
- Charm++, Cuda C
- Some applications:
	- TPACF: Distribution objects on celestial sphere
	- NAMD: Molecular dynamics
	- DSCF: Energy calculations

### Mai 17 2018

Goals:

- Lecture and analysis of the main paper.
- Write main ideas to plan the new course of the project.

**Paper: Understanding the Role of GPGPU-Accelerated SoC-Based ARM Clusters**

**Main idea:** Use a new architecture of Nvidia TX1 with 10Gb/s of network bandwidth (instead of classical 1Gb/s).

Test the performance (making an extensive test) of this architecture with several benchmarks and compare with other architectures.

ARM-Based clusters are been made for HPC and Data centers. ARM SoC's are been created with more cores. There are Mobile-Class ARM that have less CPU cores to have more GPU cores, also their put GPGPU for scientific applications (our objective).

They create Mobile-Class ARM SoC that have 1 GPGPU inside a cluster to analyze the impact of "node architecture".
   - Their objective is for Scientific applications and AI applications, more specifically Deep Learning.

They compare this cluster with Discrete GPU cluster, and conclude that have better balance between CPU/GPGPU because the energy efficiency is better.

*Be careful*: 1 ARM-Based mobile core has less power than a x86 core, that's why we always require more nodes. This implies that we require an application with **Strong Scalability**.

HPL Benchmark has good Weak Scalability.

*Tibidabo* -> Cortex A9 cores, GPU non programmable.
*MontBlanc* -> Cortex A15, GPUGPU programmed using OpenCL.


> GPGPU Model
> Normally the data between CPU and GPU has to manage by the CPU that *allocated, copied and freed* set of data. To hide this latency, GPU concurrently *transfer data* and *process new data ready*.

Also, SoC integrate this model in a same die (node, object). It reduces the latency of data send, because avoid the data send from the CPU memory to GPU memory (in this case physically are the same).

> Cuda Memory Transfer Model:
>
> 1. Host and Device Memory
>      They use different address space from host and device.
>      
> 2. Zero-Copy
>       Device threads directly to Host Memory.
>       They use PCI bus for that access to memory.
>       
> 3. Unified Memory
>      Pool of managed memory shared by host and device.
>      Access directly to the memory without any changes.

Problems with old architectures:
- There are no GPGPU (only GPU).
- There are no PCI ports for more than 1Gb/s and the CPU doesn't support more than that bandwidth.
- Programming is difficult and they don't have CUDA (Tibidabo/Montblanc).

Cluster Organization:
- 16 Jetson TX1 that have 4 Cortex A57@1.73GHz and 2 Maxwell SM.
- All connect with a CISCO 350XG with a bypassing of 120Gb/2.
- The cables are for 10 Gb/s
- Use ubuntu 14.04 in all nodes.

> Bencharks for HPC/AI Applications
>
> *HPC WorkLoads:*
>    hpl, clover leaf, tealeaf2D, tealeaf3D, jacobi.
> 
> *AI Workloads:*
>    Caffe Framework -> AlexNEt, GoogleNet
>  
>  *NAS Parallel Benchmarks (NPB):*
>     Evaluate the performance of the cluster with a Workload Class C.
>   cg, ep, it, is, mg, bt, lu, sp

> Power and Performance Measurement
>    Linux Perf  -> Performance-monitoring counters
>    nvprof        -> GPGPU events and metrics
>    nmon          -> Network traffic statistics
>    Extrae         -> Workload traces
>     
>     Metrics -> Total energy consuption and FLOPS/W

Conclusions:

- There are 3 main components that explain lower performance of the main systems
	- Branch Miss Prediction
	- NUmber of speculatevily executed instructions
	- L2 miss ratio
- AlexNet and google net performs better against any other benchmark.
- With the new architecture, the performance and the energy performance increase for 2.x and 25% respectively.
- AI application improve with this type of clusters rather than a discrete GPGPU of the same family and budget.

### Mai 22 2018

Next Steps:

- Test spack, GUIX on PC and NVIDIA
- Make the cluster and perform its tests

Deep Exascale Project
- A project that integrer Low-Medium scale code and highly scalable code in a big cluster.
- Their aim is to reach the Exascale (10^18 operations per second or 1 trillion OpxSec)
- Intel XEon Phi vs. NVIDIA Tesla
- Uses: Precision medicine, regional climate, additive manufacturing, conversion of plants to biofuels, unseen physics in materials, fundamental forces of universe.

SPACK
- Testing Spack on PC only
- Installing is easy: git clone <spack url>
- Be careful: is the folder is in /home, you have to put permissions on all folder:
```
	sudo chmod 777 /spack
```
- Spack also look for nix pkgs, be careful because don't work
- Example (installing)
```
	# spack install zlib  // Failed because look for gcc 7.3 (Nix) and show some pkg don't exist
	# spack install zlib %gcc@6  // ok installation of gcc 6.3...
```
- Always have to "connect" spack to shell (it's not permanent)
```
	$ export PATH=$SPACK_ROOT/bin:$PATH
	$ export SPACK_ROOT=<path to spack>
	$ . $SPACK_ROOT/share/spack/setup-env.sh
```
This is done each time you open a CMD window.
- Is there a way to make this permanent?

GUIX-HPC

- It's a Reproductive software deployment for HPC
- **Transactional** package manager
- Fully reproductive: package from PC = Package for HPC
- Users can create many software environments they want (isolated)

Trying to installing it had many problems, one of that said something about the move command that we just use, so we stop in this moment to try again tomorrow.

### Mai 23 2018

GUIX-HPC (part 2)

- At first again didn't work.
- But at the end works using 1 extra command.

GUIX-HPC Installation

This have some difficulties but in the end it works.

First, the best way to install these software is to follow their official tutorial:

https://www.gnu.org/software/guix/manual/html_node/Binary-Installation.html

But be careful, it has 2 other links, one in the 4th step and other in the 8th step. On 4th step is mandatory to execute all that commands, but in 8th there is 1 command you should use and thanks to another blog indicate the main command.

http://ar.to/notes/guix

 **IMPORTANT NOTE:**

**Starting in this part, i'll separate the journals of each type of software (Nix, Spack, Guix) in order to organize better the information.
I still will writing in this part of the journal but a specific points of the research**


