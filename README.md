# 🚑 Rapid Rescue AI Environment

## Overview

This project simulates a real-world emergency response system where an AI agent must decide how to allocate ambulances across multiple accident zones.

The goal is to maximize rescue effectiveness under resource constraints.

## Problem Statement

Given multiple zones with:

* Number of people affected
* Severity (risk level)
* Distance from ambulance

The agent must choose the best zone to send an ambulance.

## Observation Space

Each step provides:

### Zones

* people: number of people needing rescue
* severity: urgency level
* distance: travel cost

### Ambulances

* availability status
* busy time (if occupied)

## Action Space

* Integer index representing selected zone

Example:
"0", "1", "2"

## Reward Function

Reward is based on:

* People rescued
* Higher severity gives higher reward
* Distance penalty
* Invalid action penalty

## Environment Dynamics

* Zones worsen over time (people increase, severity rises)
* Ambulances become unavailable after dispatch
* Agent must balance urgency vs efficiency

## Tasks

* Easy: fewer zones, lower severity
* Medium: moderate complexity
* Hard: dynamic escalation with limited resources

## Setup

pip install -r server/requirements.txt

## Run Inference

python inference.py

## Deployment

This environment is deployed using:

* OpenEnv specification
* FastAPI backend
* Docker container
* Hugging Face Spaces

## API Endpoints

* /reset → initialize environment
* /step → perform action
* /health → health check

## Real-World Relevance

This environment models:

* Ambulance allocation systems
* Disaster response optimization
* Resource-constrained decision making

## Notes

Designed to evaluate AI agents in dynamic, high-stakes environments with trade-offs between speed, priority, and resource availability.

