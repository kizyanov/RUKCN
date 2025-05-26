#!/usr/bin/env python
"""RUKCN."""

import asyncio
from base64 import b64encode
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from hmac import HMAC
from hmac import new as hmac_new
from os import environ
from time import time
from typing import Any, Self
from urllib.parse import urljoin

from aiohttp import ClientConnectorError, ClientSession, ServerDisconnectedError
from dacite import (
    ForwardReferenceError,
    MissingValueError,
    StrictUnionMatchError,
    UnexpectedDataError,
    UnionMatchError,
    WrongTypeError,
    from_dict,
)
from loguru import logger
from orjson import JSONDecodeError, JSONEncodeError, dumps, loads
from result import Err, Ok, Result, do, do_async


@dataclass(frozen=True)
class TelegramSendMsg:
    """."""

    @dataclass(frozen=True)
    class Res:
        """Parse response request."""

        ok: bool


@dataclass(frozen=True)
class ApiV3ProjectListGET:
    """https://www.kucoin.com/docs-new/rest/margin-trading/credit/get-loan-market."""

    @dataclass(frozen=True)
    class Res:
        """Parse response request."""

        @dataclass(frozen=True)
        class Data:
            """."""

            currency: str
            minInterestRate: str
            marketInterestRate: str

        data: list[Data]
        code: str
        msg: str | None


@dataclass(frozen=True)
class ApiV3PurchaseOrdersGET:
    """https://www.kucoin.com/docs-new/rest/margin-trading/credit/get-purchase-orders."""

    @dataclass(frozen=True)
    class Res:
        """Parse response request."""

        @dataclass(frozen=True)
        class Data:
            """."""

            @dataclass(frozen=True)
            class Item:
                """."""

                currency: str
                purchaseOrderNo: str
                interestRate: str
                status: str

            items: list[Item]

        data: Data
        code: str
        msg: str | None


@dataclass(frozen=True)
class ApiV3LendPurchaseUpdatePOST:
    """https://www.kucoin.com/docs-new/rest/margin-trading/credit/modify-purchase."""

    @dataclass(frozen=True)
    class Res:
        """Parse response request."""

        code: str
        data: str | None
        msg: str | None


class KCN:
    """Main class collect all logic."""

    def init_envs(self: Self) -> Result[None, Exception]:
        """Init settings."""
        # All about excange
        self.KEY = self.get_env("KEY").unwrap()
        self.SECRET = self.get_env("SECRET").unwrap()
        self.PASSPHRASE = self.get_env("PASSPHRASE").unwrap()
        self.BASE_URL = self.get_env("BASE_URL").unwrap()

        # All about tlg
        self.TELEGRAM_BOT_API_KEY = self.get_env("TELEGRAM_BOT_API_KEY").unwrap()
        self.TELEGRAM_BOT_CHAT_ID = self.get_list_env("TELEGRAM_BOT_CHAT_ID").unwrap()

        logger.success("Settings are OK!")
        return Ok(None)

    def convert_to_dataclass_from_dict[T](
        self: Self,
        data_class: type[T],
        data: dict[str, Any],
    ) -> Result[T, Exception]:
        """Convert dict to dataclass."""
        try:
            return Ok(
                from_dict(
                    data_class=data_class,
                    data=data,
                ),
            )
        except (
            WrongTypeError,
            MissingValueError,
            UnionMatchError,
            StrictUnionMatchError,
            UnexpectedDataError,
            ForwardReferenceError,
        ) as exc:
            return Err(exc)

    def get_telegram_url(self: Self) -> Result[str, Exception]:
        """Get url for send telegram msg."""
        return Ok(
            f"https://api.telegram.org/bot{self.TELEGRAM_BOT_API_KEY}/sendMessage",
        )

    def get_telegram_msg(
        self: Self,
        chat_id: str,
        data: str,
    ) -> Result[dict[str, bool | str], Exception]:
        """Get msg for telegram in dict."""
        return Ok(
            {
                "chat_id": chat_id,
                "parse_mode": "HTML",
                "disable_notification": True,
                "text": data,
            },
        )

    def get_chat_ids_for_telegram(self: Self) -> Result[list[str], Exception]:
        """Get list chat id for current send."""
        return Ok(self.TELEGRAM_BOT_CHAT_ID)

    def check_telegram_response(
        self: Self,
        data: TelegramSendMsg.Res,
    ) -> Result[None, Exception]:
        """Check telegram response on msg."""
        if data.ok:
            return Ok(None)
        return Err(Exception(f"{data}"))

    async def send_msg_to_each_chat_id(
        self: Self,
        chat_ids: list[str],
        data: str,
    ) -> Result[TelegramSendMsg.Res, Exception]:
        """Send msg for each chat id."""
        method = "POST"
        for chat in chat_ids:
            await do_async(
                Ok(result)
                for telegram_url in self.get_telegram_url()
                for msg in self.get_telegram_msg(chat, data)
                for msg_bytes in self.dumps_dict_to_bytes(msg)
                for response_bytes in await self.request(
                    url=telegram_url,
                    method=method,
                    headers={
                        "Content-Type": "application/json",
                    },
                    data=msg_bytes,
                )
                for response_dict in self.parse_bytes_to_dict(response_bytes)
                for data_dataclass in self.convert_to_dataclass_from_dict(
                    TelegramSendMsg.Res,
                    response_dict,
                )
                for result in self.check_telegram_response(data_dataclass)
            )
        return Ok(TelegramSendMsg.Res(ok=False))

    async def send_telegram_msg(self: Self, data: str) -> Result[None, Exception]:
        """Send msg to telegram."""
        match await do_async(
            Ok(None)
            for chat_ids in self.get_chat_ids_for_telegram()
            for _ in await self.send_msg_to_each_chat_id(chat_ids, data)
        ):
            case Err(exc):
                logger.exception(exc)
        return Ok(None)

    def get_env(self: Self, key: str) -> Result[str, ValueError]:
        """Just get key from EVN."""
        try:
            return Ok(environ[key])
        except ValueError as exc:
            logger.exception(exc)
            return Err(exc)

    def _env_convert_to_list(self: Self, data: str) -> Result[list[str], Exception]:
        """Split str by ',' character."""
        return Ok(data.split(","))

    def get_list_env(self: Self, key: str) -> Result[list[str], Exception]:
        """Get value from ENV in list[str] format.

        in .env
        KEYS=1,2,3,4,5,6

        to
        KEYS = ['1','2','3','4','5','6']
        """
        return do(
            Ok(value_in_list)
            for value_by_key in self.get_env(key)
            for value_in_list in self._env_convert_to_list(value_by_key)
        )

    async def post_api_v3_lend_purchase_update(
        self: Self,
        data: dict[str, float | str],
    ) -> Result[ApiV3LendPurchaseUpdatePOST.Res, Exception]:
        """Modify Purchase.

        https://www.kucoin.com/docs-new/rest/margin-trading/credit/modify-purchase
        """
        uri = "/api/v3/lend/purchase/update"
        method = "POST"
        return await do_async(
            Ok(result)
            for full_url in self.get_full_url(self.BASE_URL, uri)
            for dumps_data_bytes in self.dumps_dict_to_bytes(data)
            for dumps_data_str in self.decode(dumps_data_bytes)
            for now_time in self.get_now_time()
            for data_to_sign in self.cancatinate_str(
                now_time,
                method,
                uri,
                dumps_data_str,
            )
            for headers in self.get_headers_auth(
                data_to_sign,
                now_time,
            )
            for response_bytes in await self.request(
                url=full_url,
                method=method,
                headers=headers,
                data=dumps_data_bytes,
            )
            for response_dict in self.parse_bytes_to_dict(response_bytes)
            for _ in self.logger_info(response_dict)
            for data_dataclass in self.convert_to_dataclass_from_dict(
                ApiV3LendPurchaseUpdatePOST.Res,
                response_dict,
            )
            for result in self.check_response_code(data_dataclass)
        )

    async def get_api_v3_project_list(
        self: Self,
    ) -> Result[ApiV3ProjectListGET.Res, Exception]:
        """Get Loan Market.

        https://www.kucoin.com/docs-new/rest/margin-trading/credit/get-loan-market
        """
        uri = "/api/v3/project/list"
        method = "GET"
        return await do_async(
            Ok(result)
            for params_in_url in self.get_url_params_as_str({})
            for uri_params in self.cancatinate_str(uri, params_in_url)
            for full_url in self.get_full_url(self.BASE_URL, uri_params)
            for now_time in self.get_now_time()
            for data_to_sign in self.cancatinate_str(
                now_time,
                method,
                uri_params,
            )
            for headers in self.get_headers_auth(
                data_to_sign,
                now_time,
            )
            for response_bytes in await self.request(
                url=full_url,
                method=method,
                headers=headers,
            )
            for response_dict in self.parse_bytes_to_dict(response_bytes)
            for data_dataclass in self.convert_to_dataclass_from_dict(
                ApiV3ProjectListGET.Res,
                response_dict,
            )
            for result in self.check_response_code(data_dataclass)
        )

    async def get_api_v3_purchase_orders(
        self: Self,
        params: dict[str, str | int],
    ) -> Result[ApiV3PurchaseOrdersGET.Res, Exception]:
        """Get Purchase Orders.

        https://www.kucoin.com/docs-new/rest/margin-trading/credit/get-purchase-orders
        """
        uri = "/api/v3/purchase/orders"
        method = "GET"
        return await do_async(
            Ok(result)
            for params_in_url in self.get_url_params_as_str(params)
            for uri_params in self.cancatinate_str(uri, params_in_url)
            for full_url in self.get_full_url(self.BASE_URL, uri_params)
            for now_time in self.get_now_time()
            for data_to_sign in self.cancatinate_str(
                now_time,
                method,
                uri_params,
            )
            for headers in self.get_headers_auth(
                data_to_sign,
                now_time,
            )
            for response_bytes in await self.request(
                url=full_url,
                method=method,
                headers=headers,
            )
            for response_dict in self.parse_bytes_to_dict(response_bytes)
            for _ in self.logger_info(response_dict)
            for data_dataclass in self.convert_to_dataclass_from_dict(
                ApiV3PurchaseOrdersGET.Res,
                response_dict,
            )
            for result in self.check_response_code(data_dataclass)
        )

    def get_url_params_as_str(
        self: Self,
        params: dict[str, str | int],
    ) -> Result[str, Exception]:
        """Get url params in str.

        if params is empty -> ''
        if params not empty -> ?foo=bar&zoo=net
        """
        params_in_url = "&".join([f"{key}={params[key]}" for key in sorted(params)])
        if len(params_in_url) == 0:
            return Ok("")
        return Ok("?" + params_in_url)

    def get_full_url(
        self: Self,
        base_url: str,
        next_url: str,
    ) -> Result[str, Exception]:
        """Right cancatinate base url and method url."""
        return Ok(urljoin(base_url, next_url))

    def get_headers_auth(
        self: Self,
        data: str,
        now_time: str,
    ) -> Result[dict[str, str], Exception]:
        """Get headers with encrypted data for http request."""
        return do(
            Ok(
                {
                    "KC-API-SIGN": kc_api_sign,
                    "KC-API-TIMESTAMP": now_time,
                    "KC-API-PASSPHRASE": kc_api_passphrase,
                    "KC-API-KEY": self.KEY,
                    "Content-Type": "application/json",
                    "KC-API-KEY-VERSION": "2",
                    "User-Agent": "kucoin-python-sdk/2",
                },
            )
            for secret in self.encode(self.SECRET)
            for passphrase in self.encode(self.PASSPHRASE)
            for data_in_bytes in self.encode(data)
            for kc_api_sign in self.encrypt_data(secret, data_in_bytes)
            for kc_api_passphrase in self.encrypt_data(secret, passphrase)
        )

    def convert_to_int(self: Self, data: float) -> Result[int, Exception]:
        """Convert data to int."""
        try:
            return Ok(int(data))
        except ValueError as exc:
            logger.exception(exc)
            return Err(exc)

    def get_time(self: Self) -> Result[float, Exception]:
        """Get now time as float."""
        return Ok(time())

    def get_now_time(self: Self) -> Result[str, Exception]:
        """Get now time for encrypted data."""
        return do(
            Ok(f"{time_now_in_int * 1000}")
            for time_now in self.get_time()
            for time_now_in_int in self.convert_to_int(time_now)
        )

    def check_response_code[T](
        self: Self,
        data: T,
    ) -> Result[T, Exception]:
        """Check if key `code`.

        If key `code` in dict == '200000' then success
        """
        if hasattr(data, "code") and data.code == "200000":
            return Ok(data)
        return Err(Exception(data))

    async def request(
        self: Self,
        url: str,
        method: str,
        headers: dict[str, str],
        data: bytes | None = None,
    ) -> Result[bytes, Exception]:
        """Base http request."""
        try:
            async with (
                ClientSession(
                    headers=headers,
                ) as session,
                session.request(
                    method,
                    url,
                    data=data,
                ) as response,
            ):
                res = await response.read()  # bytes
                logger.success(f"{response.status}:{method}:{url}")
                return Ok(res)
        except (ClientConnectorError, ServerDisconnectedError) as exc:
            logger.exception(exc)
            return Err(exc)

    def logger_info[T](self: Self, data: T) -> Result[T, Exception]:
        """Info logger for Pipes."""
        logger.info(data)
        return Ok(data)

    @staticmethod
    def logger_success[T](data: T) -> Result[T, Exception]:
        """Success logger for Pipes."""
        logger.success(data)
        return Ok(data)

    def cancatinate_str(self: Self, *args: str) -> Result[str, Exception]:
        """Cancatinate to str."""
        try:
            return Ok("".join(args))
        except TypeError as exc:
            logger.exception(exc)
            return Err(exc)

    def convert_bytes_to_base64(self: Self, data: bytes) -> Result[bytes, Exception]:
        """Convert bytes to base64."""
        try:
            return Ok(b64encode(data))
        except TypeError as exc:
            logger.exception(exc)
            return Err(exc)

    def encode(self: Self, data: str) -> Result[bytes, Exception]:
        """Return Ok(bytes) from str data."""
        try:
            return Ok(data.encode())
        except AttributeError as exc:
            logger.exception(exc)
            return Err(exc)

    def decode(self: Self, data: bytes) -> Result[str, Exception]:
        """Return Ok(str) from bytes data."""
        try:
            return Ok(data.decode())
        except AttributeError as exc:
            logger.exception(exc)
            return Err(exc)

    def get_default_hmac(
        self: Self,
        secret: bytes,
        data: bytes,
    ) -> Result[HMAC, Exception]:
        """Get default HMAC."""
        return Ok(hmac_new(secret, data, sha256))

    def convert_hmac_to_digest(
        self: Self,
        hmac_object: HMAC,
    ) -> Result[bytes, Exception]:
        """Convert HMAC to digest."""
        return Ok(hmac_object.digest())

    def encrypt_data(self: Self, secret: bytes, data: bytes) -> Result[str, Exception]:
        """Encript `data` to hmac."""
        return do(
            Ok(result)
            for hmac_object in self.get_default_hmac(secret, data)
            for hmac_data in self.convert_hmac_to_digest(hmac_object)
            for base64_data in self.convert_bytes_to_base64(hmac_data)
            for result in self.decode(base64_data)
        )

    def dumps_dict_to_bytes(
        self: Self,
        data: dict[str, Any],
    ) -> Result[bytes, Exception]:
        """Dumps dict to bytes[json].

        {"qaz":"edc"} -> b'{"qaz":"wsx"}'
        """
        try:
            return Ok(dumps(data))
        except JSONEncodeError as exc:
            logger.exception(exc)
            return Err(exc)

    def parse_bytes_to_dict(
        self: Self,
        data: bytes | str,
    ) -> Result[dict[str, Any], Exception]:
        """Parse bytes[json] to dict.

        b'{"qaz":"wsx"}' -> {"qaz":"wsx"}
        """
        try:
            return Ok(loads(data))
        except JSONDecodeError as exc:
            logger.exception(exc)
            return Err(exc)

    def get_best_market_rate(
        self: Self,
        data: ApiV3ProjectListGET.Res,
    ) -> Result[dict[str, str], Exception]:
        """."""
        result: dict[str, str] = {}

        for ticket in data.data:
            if ticket.marketInterestRate == ticket.minInterestRate:
                result[ticket.currency] = ticket.minInterestRate
            else:
                result[ticket.currency] = str(
                    Decimal(ticket.marketInterestRate) - Decimal("0.0001"),
                )

        return Ok(result)

    async def get_all_purchase(
        self: Self,
    ) -> Result[list[ApiV3PurchaseOrdersGET.Res.Data.Item], Exception]:
        """."""
        result = []
        current_page = 1

        while True:
            match await do_async(
                Ok(purchase)
                for purchase in await self.get_api_v3_purchase_orders(
                    {
                        "status": "PENDING",
                        "currentPage": current_page,
                    },
                )
            ):
                case Ok(purchase):
                    if len(purchase.data.items) == 0:
                        break
                    current_page += 1
                    result += purchase.data.items

                case Err(exc):
                    logger.exception(exc)
                    break
        return Ok(result)

    async def compare_market_rate(
        self: Self,
        my_purchase: list[ApiV3PurchaseOrdersGET.Res.Data.Item],
        best_market_rate: dict[str, str],
    ) -> Result[None, Exception]:
        """."""
        for purchase in my_purchase:
            if (
                purchase.currency in best_market_rate
                and purchase.interestRate != best_market_rate[purchase.currency]
            ):
                msg = f"Need update rate:{purchase.currency} from {purchase.interestRate} to {best_market_rate[purchase.currency]}"
                logger.info(msg)

                match await do_async(
                    Ok(_)
                    for _ in await self.post_api_v3_lend_purchase_update(
                        {
                            "currency": purchase.currency,
                            "purchaseOrderNo": purchase.purchaseOrderNo,
                            "interestRate": best_market_rate[purchase.currency],
                        },
                    )
                    for _ in await self.send_telegram_msg(
                        f"Update rate:{purchase.currency} from {purchase.interestRate} to {best_market_rate[purchase.currency]}",
                    )
                    for _ in KCN.logger_success(
                        f"Update rate:{purchase.currency} from {purchase.interestRate} to {best_market_rate[purchase.currency]}",
                    )
                ):
                    case Err(exc):
                        logger.exception(exc)
        return Ok(None)

    async def change_rate_margin(self: Self) -> Result[None, Exception]:
        """."""
        match await do_async(
            Ok(_)
            for project_list in await self.get_api_v3_project_list()
            for best_market_rate in self.get_best_market_rate(project_list)
            for all_purchase in await self.get_all_purchase()
            for _ in await self.compare_market_rate(
                all_purchase,
                best_market_rate,
            )
        ):
            case Err(exc):
                logger.exception(exc)
        return Ok(None)

    async def infinity_task(self: Self) -> Result[None, Exception]:
        """Infinity run tasks."""
        while True:
            async with asyncio.TaskGroup() as tg:
                await tg.create_task(self.change_rate_margin())
            await asyncio.sleep(60 * 60)

        return Ok(None)


async def main() -> Result[None, Exception]:
    """Collect of major func."""
    kcn = KCN()
    match await do_async(
        Ok(None)
        for _ in kcn.init_envs()
        for _ in KCN.logger_success("Pre-init OK!")
        for _ in await kcn.send_telegram_msg("KuCoin settings are OK!")
        for _ in await kcn.infinity_task()
    ):
        case Ok(None):
            pass
        case Err(exc):
            logger.exception(exc)
            return Err(exc)
    return Ok(None)


if __name__ == "__main__":
    """Main enter."""
    asyncio.run(main())
