from openmm import MonteCarloBarostat, unit
from openmm.app import Simulation
from openmm.unit import Quantity


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

        self.simulation = simulation
        self.temperature = temperature
        self.pressure = pressure
        self.time = time
        self.name = name

        timestep = simulation.integrator.getStepSize()
        self.steps = int(round(time / timestep))

    def run(self, frequency=500):
        """
        Executes the molecular dynamics stage.

        This method sets the new temperature, initializes velocities,
        configures the barostat (if pressure is not None), and runs the steps.

        Args:
            frequency: The frequency (in steps) for the Monte Carlo Barostat moves.
                Only used if self.pressure is not None. Defaults to 500.
        """

        print(f"\n=== Starting stage {self.name} ===")
        print(f"Temperature: {self.temperature}, Pressure: {self.pressure}")
        print(f"Time: {self.time}")

        self.simulation.integrator.setTemperature(self.temperature)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        self._set_barostat(frequency)
        self.simulation.context.reinitialize(preserveState=True)
        self.simulation.step(self.steps)

        print(f"Completed stage {self.name}")

    def _set_barostat(self, frequency: int):
        """
        Removes any existing MonteCarloBarostat and adds a new one if
        self.pressure is not None.

        Args:
            frequency: The frequency for the barostat moves.
        """

        system = self.simulation.system

        for i, force in enumerate(list(system.getForces())):
            if isinstance(force, MonteCarloBarostat):
                system.removeForce(i)

        if self.pressure is not None:
            system.addForce(
                MonteCarloBarostat(self.pressure, self.temperature, frequency)
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
    ):
        """
        Initializes the protocol manager and generates the schedule.

        Args:
            simulation: The OpenMM Simulation object to be used for all steps.
            max_pressure: The maximum pressure to be used in the ramping stages
                (md9). Defaults to 50,000 bar.
            max_temperature: The maximum temperature for the equilibration. Defaults to 600 K.

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

        self.simulation = simulation
        self.schedule: list[dict] = []
        self._generate_schedule(max_pressure)

    def _generate_schedule(
        self, max_pressure: Quantity, max_temperature: Quantity = 600 * unit.kelvin
    ):
        """
        Generates the 21-stage pressure ramping schedule based on a
        maximum pressure value.

        Args:
            max_pressure: The peak pressure value used to scale other pressure steps.
            max_temperature: The maximum temperature for the equilibration. Defaults to 600 K.
        """

        self.schedule = [
            {
                "temperature": max_temperature,
                "pressure": None,
                "time": 50 * unit.picosecond,
                "name": "md1",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": None,
                "time": 50 * unit.picosecond,
                "name": "md2",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": max_pressure * 0.02,
                "time": 50 * unit.picosecond,
                "name": "md3",
            },
            {
                "temperature": max_temperature,
                "pressure": None,
                "time": 50 * unit.picosecond,
                "name": "md4",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": None,
                "time": 100 * unit.picosecond,
                "name": "md5",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": max_pressure * 0.6,
                "time": 50 * unit.picosecond,
                "name": "md6",
            },
            {
                "temperature": max_temperature,
                "pressure": None,
                "time": 50 * unit.picosecond,
                "name": "md7",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": None,
                "time": 100 * unit.picosecond,
                "name": "md8",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": max_pressure,
                "time": 50 * unit.picosecond,
                "name": "md9",
            },
            {
                "temperature": max_temperature,
                "pressure": None,
                "time": 50 * unit.picosecond,
                "name": "md10",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": None,
                "time": 100 * unit.picosecond,
                "name": "md11",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": max_pressure * 0.5,
                "time": 5 * unit.picosecond,
                "name": "md12",
            },
            {
                "temperature": max_temperature,
                "pressure": None,
                "time": 5 * unit.picosecond,
                "name": "md13",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": None,
                "time": 10 * unit.picosecond,
                "name": "md14",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": max_pressure * 0.1,
                "time": 5 * unit.picosecond,
                "name": "md15",
            },
            {
                "temperature": max_temperature,
                "pressure": None,
                "time": 5 * unit.picosecond,
                "name": "md16",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": None,
                "time": 10 * unit.picosecond,
                "name": "md17",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": max_pressure * 0.01,
                "time": 5 * unit.picosecond,
                "name": "md18",
            },
            {
                "temperature": max_temperature,
                "pressure": None,
                "time": 5 * unit.picosecond,
                "name": "md19",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": None,
                "time": 10 * unit.picosecond,
                "name": "md20",
            },
            {
                "temperature": 300 * unit.kelvin,
                "pressure": 1 * unit.bar,
                "time": 800 * unit.picosecond,
                "name": "md21",
            },
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

        if not self.schedule:
            raise RuntimeError(
                "Schedule is empty. Probably an error ocured during schedule generation."
            )

        print(f"\n--- Protocol Starting: {len(self.schedule)} Stages ---")

        for task in self.schedule:
            step = MDStep(simulation=self.simulation, **task)
            step.run(frequency=barostat_frequency)

        print("\n--- Protocol Completed Successfully ---")
