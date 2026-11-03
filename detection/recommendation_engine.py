
class RecommendationEngine:
    """
    Provides security recommendations based on AI-NIDS predictions.
    This engine recommends defensive actions only.
    It does not modify firewall or network settings.
    """

    RECOMMENDATIONS = {
        "BENIGN": {
            "severity": "Informational",
            "description": "No malicious activity was identified by the model.",
            "risk": "No immediate threat indicated by this prediction.",
            "actions": [
                "Continue monitoring network traffic.",
                "Maintain updated security software.",
                "Review unusual traffic patterns periodically."
            ]
        },

        "Botnet": {
            "severity": "Critical",
            "description": "Traffic resembles communication associated with a botnet.",
            "risk": "Compromised devices may communicate with command-and-control servers.",
            "actions": [
                "Investigate the source and destination IP addresses.",
                "Check the affected device for malware.",
                "Review unusual outbound connections and DNS requests.",
                "Update endpoint security software and scan the device."
            ]
        },

        "BruteForce": {
            "severity": "High",
            "description": "Repeated login attempts may indicate password guessing.",
            "risk": "An attacker may gain unauthorized account access.",
            "actions": [
                "Review authentication and login logs.",
                "Enable multi-factor authentication.",
                "Use strong password policies.",
                "Consider account lockout and login rate limiting."
            ]
        },

        "DDoS": {
            "severity": "Critical",
            "description": "Traffic resembles a distributed denial-of-service attack.",
            "risk": "High traffic volume may make a service unavailable.",
            "actions": [
                "Monitor incoming traffic volume and source distribution.",
                "Consider rate limiting at the network edge.",
                "Review firewall and router configurations.",
                "Consider using a trusted DDoS protection service."
            ]
        },

        "DoS": {
            "severity": "High",
            "description": "Traffic resembles a denial-of-service attack.",
            "risk": "A targeted service may become slow or unavailable.",
            "actions": [
                "Identify the targeted service and affected destination.",
                "Review traffic spikes and connection patterns.",
                "Consider rate limiting suspicious traffic.",
                "Monitor server resource usage."
            ]
        },

        "PortScan": {
            "severity": "Medium",
            "description": "Traffic resembles scanning of network ports.",
            "risk": "An attacker may be searching for exposed services.",
            "actions": [
                "Review the source IP and scanned destination ports.",
                "Close unnecessary open ports.",
                "Restrict access to administrative services.",
                "Review firewall rules and network logs."
            ]
        },

        "WebAttack": {
            "severity": "High",
            "description": "Traffic resembles a web-based attack.",
            "risk": "Web applications may be exposed to unauthorized access or exploitation.",
            "actions": [
                "Review web server and application logs.",
                "Validate and sanitize user inputs.",
                "Keep web frameworks and server software updated.",
                "Consider a properly configured web application firewall."
            ]
        }
    }

    @classmethod
    def get_recommendation(cls, attack_type):
        attack_type = str(attack_type).strip()

        recommendation = cls.RECOMMENDATIONS.get(attack_type)

        if recommendation is None:
            return {
                "attack_type": attack_type,
                "severity": "Unknown",
                "description": "The predicted class has no configured recommendation.",
                "risk": "Risk could not be assessed.",
                "actions": [
                    "Review the prediction and available network evidence.",
                    "Investigate the source and destination IP addresses.",
                    "Consult network security logs."
                ]
            }

        return {
            "attack_type": attack_type,
            **recommendation
        }


if __name__ == "__main__":
    engine = RecommendationEngine()

    test_attacks = [
        "BENIGN",
        "Botnet",
        "BruteForce",
        "DDoS",
        "DoS",
        "PortScan",
        "WebAttack"
    ]

    for attack in test_attacks:
        result = engine.get_recommendation(attack)

        print("\n" + "=" * 50)
        print("Attack:", result["attack_type"])
        print("Severity:", result["severity"])
        print("Description:", result["description"])
        print("Risk:", result["risk"])
        print("Recommended Actions:")

        for action in result["actions"]:
            print("-", action)

