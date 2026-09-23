from dataclasses import dataclass
from pathlib import Path

from openmm import (
    MonteCarloAnisotropicBarostat,
    MonteCarloBarostat,
    MonteCarloMembraneBarostat,
    unit,
)
from openmm.app import Simulation, StateDataReporter
from openmm.unit import Quantity


@dataclass(frozen=True)
class Stage:
    """Thermodynamic conditions and duration of one protocol stage."""

    temperature: Quantity
    pressure: Quantity | None
    time: Quantity
    name: str


BAROSTAT_TYPES = (
    MonteCarloBarostat,
    MonteCarloAnisotropicBarostat,
    MonteCarloMembraneBarostat,
)


class MDStep:
    """
    Represents a single molecular dynamics (MD) stage with defined
    thermodynamic conditions (T/P/time).

    It manages the setup and execution of the simulation step,
    including temperature and pressure control (Barostat).
    """

    def __init__(
        self,
        simulation: Simulation,
        temperature: Quantity,
        pressure: Quantity,
        time: Quantity,
        name: str,
        output_dir: Path | None = None,
    ):
        """
        Initializes an MD step configuration.

        Args:
            simulation: The OpenMM Simulation object to be run.
            temperature: The target temperature for this stage (e.g., 300*unit.kelvin).
            pressure: The target pressure for this stage (e.g., 1*unit.bar).
                If None, no barostat will be applied (NVT ensemble).
            time: The total duration of the stage (e.g., 50*unit.picosecond).
            name: A descriptive name for the stage (e.g., 'equilibration_NPT').

        Raises:
            TypeError: If any argument type is incorrect.
        """

        if any(not isinstance(arg, Quantity) for arg in [temperature, time]):
            raise TypeError(
                "Arguments 'temperature' and 'time' should be instances of openmm.unit.Quantity"
            )

        if not isinstance(simulation, Simulation):
            raise TypeError(
                "Argument 'simulation' should be an instance of openmm.app.Simulation"
            )

        if pressure is not None and not isinstance(pressure, Quantity):
            raise TypeError(
                "Argument 'pressure' should be an instance of openmm.unit.Quantity or None"
            )

        if not isinstance(name, str):
            raise TypeError("Argument 'name' should be an instance of str")

        if output_dir is not None and not isinstance(output_dir, Path):
            raise TypeError("Argument 'output_dir' should be a pathlib.Path or None")

        self.simulation = simulation
        self.temperature = temperature
        self.pressure = pressure
        self.time = time
        self.name = name
        self.output_dir = output_dir

        timestep = simulation.integrator.getStepSize()
        self.steps = int(round(time / timestep))

    def run(self, frequency=500):
        """
        Executes the molecular dynamics stage.

        This method sets the new target temperature, configures the barostat
        (if pressure is not None), and runs the steps.

        Velocities are intentionally not reinitialized between stages. The
        protocol uses abrupt changes of the thermostat target temperature, but
        re-sampling velocities at every stage would discard the dynamical
        history of the preceding stage.

        Args:
            frequency: The frequency (in steps) for the Monte Carlo Barostat moves.
                Only used if self.pressure is not None. Defaults to 500.
        """

        print(f"\n=== Starting stage {self.name} ===")
        print(f"Temperature: {self.temperature}, Pressure: {self.pressure}")
        print(f"Time: {self.time}")

        self.simulation.integrator.setTemperature(self.temperature)
        self._set_barostat(frequency)
        self.simulation.context.reinitialize(preserveState=True)
        reporter = None
        if self.output_dir is not None:
            reporter = StateDataReporter(
                str(self.output_dir / f"{self.name}.csv"),
                max(1, min(frequency, self.steps)),
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                separator=",",
            )
            self.simulation.reporters.append(reporter)

        try:
            self.simulation.step(self.steps)
            if self.output_dir is not None:
                self.simulation.saveCheckpoint(str(self.output_dir / f"{self.name}.chk"))
        finally:
            if reporter is not None:
                self.simulation.reporters.remove(reporter)

        print(f"Completed stage {self.name}")

    def _set_barostat(self, frequency: int):
        """
        Removes any existing MonteCarloBarostat and adds a new one if
        self.pressure is not None.

        Args:
            frequency: The frequency for the barostat moves.
        """

        system = self.simulation.system

        barostat_indices = [
            i
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), BAROSTAT_TYPES)
        ]
        for i in reversed(barostat_indices):
            system.removeForce(i)

        remaining = [
            force
            for force in system.getForces()
            if isinstance(force, BAROSTAT_TYPES)
        ]
        if remaining:
            raise RuntimeError("Failed to remove all existing Monte Carlo barostats")

        if self.pressure is not None:
            system.addForce(
                MonteCarloBarostat(self.pressure, self.temperature, frequency)
            )

        added = [
            force
            for force in system.getForces()
            if isinstance(force, BAROSTAT_TYPES)
        ]
        expected = 1 if self.pressure is not None else 0
        if len(added) != expected:
            raise RuntimeError(
                f"Expected {expected} Monte Carlo barostat(s), found {len(added)}"
            )


class TwentyOneStepProtocol:
    """
    Manages the creation and execution of the specific 21-stage Molecular Dynamics
    equilibration protocol from Larsen et al. (2011).

    This protocol is designed for the rigorous equilibration of complex, dense systems
    (like polymer melts or glasses) using pressure ramping and temperature cycling to
    thoroughly sample the phase space and achieve structural stability.

    Reference:
        Larsen GS, Lin P, Hart KE, Colina CM (2011) Macromolecules 44:6944–6951.
    """

    def __init__(
        self,
        simulation: Simulation,
        max_pressure: Quantity = 50_000 * unit.bar,
        max_temperature: Quantity = 600 * unit.kelvin,
        target_temperature: Quantity = 300 * unit.kelvin,
        target_pressure: Quantity = 1 * unit.bar,
        output_dir: str | Path | None = None,
    ):
        """
        Initializes the protocol manager and generates the schedule.

        Args:
            simulation: The OpenMM Simulation object to be used for all steps.
            max_pressure: The maximum pressure to be used in the ramping stages
                (md9). Defaults to 50,000 bar.
            max_temperature: The maximum temperature for the equilibration. Defaults to 600 K.
            target_temperature: The cooling and final equilibration temperature. Defaults to 300 K.
            target_pressure: The final pressure at md21. Defaults to 1 bar.
            output_dir: Directory for per-stage CSV diagnostics and checkpoints.

        Raises:
            TypeError: If argument types are incorrect.
        """

        if not isinstance(simulation, Simulation):
            raise TypeError(
                "Argument 'simulation' should be an instance of openmm.app.Simulation"
            )

        if not isinstance(max_pressure, Quantity):
            raise TypeError(
                "Argument 'max_pressure' should be an instance of openmm.unit.Quantity"
            )

        if not isinstance(max_temperature, Quantity):
            raise TypeError(
                "Argument 'max_temperature' should be an instance of openmm.unit.Quantity"
            )

        if not isinstance(target_temperature, Quantity):
            raise TypeError(
                "Argument 'target_temperature' should be an instance of openmm.unit.Quantity"
            )

        if not isinstance(target_pressure, Quantity):
            raise TypeError(
                "Argument 'target_pressure' should be an instance of openmm.unit.Quantity"
            )

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for name, value in (
            ("max_pressure", max_pressure),
            ("max_temperature", max_temperature),
            ("target_temperature", target_temperature),
            ("target_pressure", target_pressure),
        ):
            if value <= 0 * value.unit:
                raise ValueError(f"Argument '{name}' must be positive")

        if not hasattr(simulation.integrator, "setTemperature"):
            raise TypeError(
                "The simulation integrator must support setTemperature()"
            )

        if not simulation.system.usesPeriodicBoundaryConditions():
            raise ValueError(
                "The system must use periodic boundary conditions for this protocol"
            )

        self.simulation = simulation
        self.output_dir = output_dir
        self.schedule: list[Stage] = []
        self._generate_schedule(
            max_pressure, max_temperature, target_temperature, target_pressure
        )

    def _generate_schedule(
        self,
        max_pressure: Quantity,
        max_temperature: Quantity = 600 * unit.kelvin,
        target_temperature: Quantity = 300 * unit.kelvin,
        target_pressure: Quantity = 1 * unit.bar,
    ):
        """
        Generates the 21-stage pressure ramping schedule based on a
        maximum pressure value.

        Args:
            max_pressure: The peak pressure value used to scale other pressure steps.
            max_temperature: The maximum temperature for the equilibration. Defaults to 600 K.
            target_temperature: The cooling and final equilibration temperature. Defaults to 300 K.
            target_pressure: The final pressure. Defaults to 1 bar.
        """

        self.schedule = [
            Stage(
                temperature= max_temperature,
                pressure= None,
                time= 50 * unit.picosecond,
                name= "md1",
            ),
            Stage(
                temperature= target_temperature,
                pressure= None,
                time= 50 * unit.picosecond,
                name= "md2",
            ),
            Stage(
                temperature= target_temperature,
                pressure= max_pressure * 0.02,
                time= 50 * unit.picosecond,
                name= "md3",
            ),
            Stage(
                temperature= max_temperature,
                pressure= None,
                time= 50 * unit.picosecond,
                name= "md4",
            ),
            Stage(
                temperature= target_temperature,
                pressure= None,
                time= 100 * unit.picosecond,
                name= "md5",
            ),
            Stage(
                temperature= target_temperature,
                pressure= max_pressure * 0.6,
                time= 50 * unit.picosecond,
                name= "md6",
            ),
            Stage(
                temperature= max_temperature,
                pressure= None,
                time= 50 * unit.picosecond,
                name= "md7",
            ),
            Stage(
                temperature= target_temperature,
                pressure= None,
                time= 100 * unit.picosecond,
                name= "md8",
            ),
            Stage(
                temperature= target_temperature,
                pressure= max_pressure,
                time= 50 * unit.picosecond,
                name= "md9",
            ),
            Stage(
                temperature= max_temperature,
                pressure= None,
                time= 50 * unit.picosecond,
                name= "md10",
            ),
            Stage(
                temperature= target_temperature,
                pressure= None,
                time= 100 * unit.picosecond,
                name= "md11",
            ),
            Stage(
                temperature= target_temperature,
                pressure= max_pressure * 0.5,
                time= 5 * unit.picosecond,
                name= "md12",
            ),
            Stage(
                temperature= max_temperature,
                pressure= None,
                time= 5 * unit.picosecond,
                name= "md13",
            ),
            Stage(
                temperature= target_temperature,
                pressure= None,
                time= 10 * unit.picosecond,
                name= "md14",
            ),
            Stage(
                temperature= target_temperature,
                pressure= max_pressure * 0.1,
                time= 5 * unit.picosecond,
                name= "md15",
            ),
            Stage(
                temperature= max_temperature,
                pressure= None,
                time= 5 * unit.picosecond,
                name= "md16",
            ),
            Stage(
                temperature= target_temperature,
                pressure= None,
                time= 10 * unit.picosecond,
                name= "md17",
            ),
            Stage(
                temperature= target_temperature,
                pressure= max_pressure * 0.01,
                time= 5 * unit.picosecond,
                name= "md18",
            ),
            Stage(
                temperature= max_temperature,
                pressure= None,
                time= 5 * unit.picosecond,
                name= "md19",
            ),
            Stage(
                temperature= target_temperature,
                pressure= None,
                time= 10 * unit.picosecond,
                name= "md20",
            ),
            Stage(
                temperature= target_temperature,
                pressure= target_pressure,
                time= 800 * unit.picosecond,
                name= "md21",
            ),
        ]

    def run(self, barostat_frequency: int = 500):
        """
        Executes all MD stages defined in the internal schedule using the
        MDStep executor class.

        Args:
            barostat_frequency: The frequency for the Monte Carlo Barostat moves.
                Defaults to 500.

        Raises:
            TypeError: If barostat_frequency is not an integer.
            RuntimeError: If the schedule list is empty (should not happen after init).
        """

        if not isinstance(barostat_frequency, int):
            raise TypeError(
                "Argument 'barostat_frequency' should be an instance of int"
            )
        if barostat_frequency <= 0:
            raise ValueError("Argument 'barostat_frequency' must be positive")

        if not self.schedule:
            raise RuntimeError(
                "Schedule is empty. Probably an error ocured during schedule generation."
            )

        print(f"\n--- Protocol Starting: {len(self.schedule)} Stages ---")

        for task in self.schedule:
            step = MDStep(
                simulation=self.simulation,
                temperature=task.temperature,
                pressure=task.pressure,
                time=task.time,
                name=task.name,
                output_dir=self.output_dir,
            )
            step.run(frequency=barostat_frequency)

        print("\n--- Protocol Completed Successfully ---")
