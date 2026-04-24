#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_pins(raw_pins: str) -> list[int]:
    parts = [p.strip() for p in raw_pins.split(",") if p.strip()]
    if not parts:
        raise ValueError("No GPIO pins provided.")

    pins: list[int] = []
    for part in parts:
        pin = int(part)
        if pin < 0:
            raise ValueError(f"Invalid GPIO pin number: {pin}")
        pins.append(pin)

    # Keep user order but drop duplicates.
    return list(dict.fromkeys(pins))


class LedPwmController:
    def set_brightness(self, value: float) -> None:
        raise NotImplementedError

    def off(self) -> None:
        self.set_brightness(0.0)

    def close(self) -> None:
        raise NotImplementedError


class GpioZeroController(LedPwmController):
    def __init__(self, pins: Sequence[int], frequency_hz: float):
        from gpiozero import PWMLED

        self._leds = [PWMLED(pin=pin, frequency=frequency_hz, initial_value=0.0) for pin in pins]

    def set_brightness(self, value: float) -> None:
        level = clamp01(value)
        for led in self._leds:
            led.value = level

    def close(self) -> None:
        for led in self._leds:
            led.off()
            led.close()


class RPiGPIOController(LedPwmController):
    def __init__(self, pins: Sequence[int], frequency_hz: float):
        import RPi.GPIO as GPIO

        self._GPIO = GPIO
        self._pins = list(pins)
        self._pwm_channels = []

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        for pin in self._pins:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, float(frequency_hz))
            pwm.start(0.0)
            self._pwm_channels.append(pwm)

    def set_brightness(self, value: float) -> None:
        duty = clamp01(value) * 100.0
        for pwm in self._pwm_channels:
            pwm.ChangeDutyCycle(duty)

    def close(self) -> None:
        for pwm in self._pwm_channels:
            pwm.ChangeDutyCycle(0.0)
            pwm.stop()
        # Cleanup only the pins touched by this script.
        self._GPIO.cleanup(self._pins)


def build_controller(pins: Sequence[int], frequency_hz: float, backend: str) -> tuple[LedPwmController, str]:
    errors: list[str] = []

    def _try_gpiozero() -> LedPwmController:
        return GpioZeroController(pins=pins, frequency_hz=frequency_hz)

    def _try_rpigpio() -> LedPwmController:
        return RPiGPIOController(pins=pins, frequency_hz=frequency_hz)

    candidates: list[tuple[str, Callable[[], LedPwmController]]]
    if backend == "gpiozero":
        candidates = [("gpiozero", _try_gpiozero)]
    elif backend == "rpi":
        candidates = [("RPi.GPIO", _try_rpigpio)]
    else:
        # On Raspberry Pi 5, gpiozero with lgpio backend is generally more reliable.
        candidates = [("gpiozero", _try_gpiozero), ("RPi.GPIO", _try_rpigpio)]

    for name, factory in candidates:
        try:
            return factory(), name
        except Exception as exc:  # pragma: no cover - depends on local Pi setup
            errors.append(f"{name}: {exc}")

    message = [
        "Failed to initialize any GPIO backend.",
        "Tried backends:",
        *[f"  - {line}" for line in errors],
        "",
        "Install one of the following on Raspberry Pi OS:",
        "  sudo apt install -y python3-gpiozero python3-rpi.gpio python3-rpi-lgpio",
    ]
    raise RuntimeError("\n".join(message))


@dataclass
class RunConfig:
    pins: list[int]
    mode: str
    backend: str
    frequency_hz: float
    brightness: float
    min_brightness: float
    max_brightness: float
    step: float
    interval_s: float
    blink_on_s: float
    blink_off_s: float
    cycles: int


def iter_sweep_levels(min_v: float, max_v: float, step: float) -> Iterable[float]:
    low = clamp01(min_v)
    high = clamp01(max_v)
    if high < low:
        low, high = high, low

    step = max(1e-4, abs(step))
    span = max(0.0, high - low)
    if span <= 1e-6:
        while True:
            yield low

    points = max(2, int(math.ceil(span / step)) + 1)
    up = [low + span * i / (points - 1) for i in range(points)]
    down = list(reversed(up[1:-1]))
    pattern = up + down

    while True:
        for value in pattern:
            yield value


def run_pattern(controller: LedPwmController, config: RunConfig) -> None:
    mode = config.mode
    cycles = config.cycles

    if mode == "set":
        controller.set_brightness(config.brightness)
        print(f"Set brightness to {clamp01(config.brightness):.2f}; press Ctrl+C to exit.")
        while True:
            time.sleep(1.0)

    if mode == "blink":
        print("Blink mode started; press Ctrl+C to stop.")
        count = 0
        while cycles <= 0 or count < cycles:
            controller.set_brightness(config.max_brightness)
            time.sleep(config.blink_on_s)
            controller.set_brightness(config.min_brightness)
            time.sleep(config.blink_off_s)
            count += 1
        return

    if mode == "breathe":
        print("Breathe mode started; press Ctrl+C to stop.")
        level_stream = iter_sweep_levels(config.min_brightness, config.max_brightness, config.step)
        count = 0
        current_phase_up = True
        prev_level: float | None = None

        for level in level_stream:
            controller.set_brightness(level)
            time.sleep(config.interval_s)

            if prev_level is not None:
                if current_phase_up and level < prev_level:
                    current_phase_up = False
                elif (not current_phase_up) and level > prev_level:
                    current_phase_up = True
                    count += 1
                    if cycles > 0 and count >= cycles:
                        break
            prev_level = level
        return

    raise ValueError(f"Unsupported mode: {mode}")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Raspberry Pi 5 GPIO LED/fill-light test with software PWM. "
            "Default GPIO pins are 26 and 16 (BCM numbering)."
        )
    )
    parser.add_argument("--pins", type=str, default="26,16", help="BCM GPIO pins, comma-separated. Default: 26,16")
    parser.add_argument("--backend", choices=["auto", "gpiozero", "rpi"], default="auto", help="GPIO backend selection")
    parser.add_argument("--mode", choices=["set", "blink", "breathe"], default="breathe", help="LED test pattern mode")
    parser.add_argument("--frequency", type=float, default=200.0, help="PWM frequency in Hz")
    parser.add_argument("--brightness", type=float, default=0.6, help="Brightness for set mode, range 0.0~1.0")
    parser.add_argument("--min", dest="min_brightness", type=float, default=0.05, help="Minimum brightness for blink/breathe")
    parser.add_argument("--max", dest="max_brightness", type=float, default=0.8, help="Maximum brightness for blink/breathe")
    parser.add_argument("--step", type=float, default=0.03, help="Breathe brightness step")
    parser.add_argument("--interval", type=float, default=0.03, help="Breathe step interval seconds")
    parser.add_argument("--blink-on", type=float, default=0.20, help="Blink ON duration seconds")
    parser.add_argument("--blink-off", type=float, default=0.20, help="Blink OFF duration seconds")
    parser.add_argument("--cycles", type=int, default=0, help="Run cycles; 0 means infinite")
    args = parser.parse_args()

    return RunConfig(
        pins=parse_pins(args.pins),
        mode=args.mode,
        backend=args.backend,
        frequency_hz=max(1.0, float(args.frequency)),
        brightness=clamp01(args.brightness),
        min_brightness=clamp01(args.min_brightness),
        max_brightness=clamp01(args.max_brightness),
        step=max(1e-4, float(args.step)),
        interval_s=max(1e-3, float(args.interval)),
        blink_on_s=max(1e-3, float(args.blink_on)),
        blink_off_s=max(1e-3, float(args.blink_off)),
        cycles=int(args.cycles),
    )


def main() -> int:
    config = parse_args()
    controller, backend_name = build_controller(config.pins, config.frequency_hz, config.backend)

    stop_requested = False

    def _handle_signal(signum: int, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}, shutting down...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("=" * 56)
    print("Raspberry Pi LED Fill Light Test")
    print(f"Backend      : {backend_name}")
    print(f"GPIO pins    : {config.pins} (BCM)")
    print(f"PWM frequency: {config.frequency_hz:.1f} Hz")
    print(f"Mode         : {config.mode}")
    print("Press Ctrl+C to stop.")
    print("=" * 56)

    try:
        run_pattern(controller, config)
    except KeyboardInterrupt:
        stop_requested = True
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        controller.off()
        controller.close()
        if stop_requested:
            print("GPIO cleaned up.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
