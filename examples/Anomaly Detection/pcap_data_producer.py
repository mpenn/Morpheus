#!/usr/bin/python3

import click
import json

from tqdm import tqdm
from faker import Faker
from os import path
from random import randint

faker = Faker()


def generate_random_digits(n):
    range_start = 10 ** (n - 1)
    range_end = (10 ** n) - 1
    return randint(range_start, range_end)


def generate_json():
    row_dict = {}
    row_dict["timestamp"] = str(faker.unix_time() * 1000000)
    row_dict["host_ip"] = faker.ipv4()
    row_dict["data_len"] = str(generate_random_digits(2))
    data = "GET /latest/meta-data/network/interfaces/macs/ HTTP/1.1\r\nHost: {}\r\nUser-Agent: aws-sdk-go/1.16.26 (go1.13.15; linux; amd64)\r\nAccept-Encoding: gzip\r\n\r\n".format(
        row_dict["host_ip"]
    )
    row_dict["data"] = data
    row_dict["src_mac"] = faker.mac_address()
    row_dict["dest_mac"] = faker.mac_address()
    row_dict["protocol"] = "6"
    row_dict["src_ip"] = faker.ipv4()
    row_dict["dest_ip"] = faker.ipv4()
    row_dict["src_port"] = str(generate_random_digits(5))
    row_dict["dest_port"] = str(generate_random_digits(2))
    row_dict["flags"] = str(generate_random_digits(1))
    json_row = json.dumps(row_dict)
    return json_row


def write_to_file(file, count):
    for i in tqdm(range(count)):
        file.write(generate_json())
        file.write("\n")
    file.close()


@click.command()
@click.option("--count", default=1000, help="The number of logs that must be produced")
@click.option(
    "--file",
    default="pcap.jsonlines",
    help="The path to the file where the created logs will be saved.",
)
def pcap_data_producer(count, file):
    if path.isdir(file):
        raise Exception("Please only pass file paths as the given path is a directory.")
    if path.exists(file):
        raise Exception("File {} already exists".format(file))
    file = open(file, "w")
    write_to_file(file, count)
    print("Logs were produced and saved to '{}'".format(file.name))


if __name__ == "__main__":
    pcap_data_producer()