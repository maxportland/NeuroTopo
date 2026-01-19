"""
AI-powered visual quality assessment for retopologized meshes.

Uses vision models to evaluate mesh topology quality by analyzing renders
and providing specific, actionable feedback.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

from neurotopo.core.mesh import Mesh
from neurotopo.utils.keychain import get_openai_api_key, get_anthropic_api_key

logger = logging.getLogger("neurotopo.evaluation.ai_quality")


class IssueSeverity(Enum):
    """Severity levels for topology issues."""
    INFO = "info"  # Minor suggestion
    WARNING = "warning"  # Should fix for production
    ERROR = "error"  # Significant problem
    CRITICAL = "critical"  # Breaks functionality


class IssueCategory(Enum):
    """Categories of topology issues."""
    EDGE_FLOW = "edge_flow"
    POLE_PLACEMENT = "pole_placement"
    QUAD_QUALITY = "quad_quality"
    DENSITY = "density"
    DEFORMATION = "deformation"
    SILHOUETTE = "silhouette"
    SYMMETRY = "symmetry"
    GENERAL = "general"


@dataclass
class TopologyIssue:
    """A specific topology issue identified by AI."""
    category: IssueCategory
    severity: IssueSeverity
    description: str
    location_hint: Optional[str] = None  # e.g., "left eye area"
    bbox_2d: Optional[list[float]] = None  # [left, top, right, bottom] normalized
    view_index: Optional[int] = None
    recommendation: Optional[str] = None


@dataclass
class AIQualityReport:
    """Complete AI quality assessment report."""
    overall_score: float  # 0-100
    issues: list[TopologyIssue] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    summary: str = ""
    mesh_type_detected: Optional[str] = None
    animation_ready: bool = False
    recommended_actions: list[str] = field(default_factory=list)
    
    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)
    
    @property
    def has_errors(self) -> bool:
        return any(i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL) 
                   for i in self.issues)
    
    @property
    def issue_count_by_severity(self) -> dict[str, int]:
        counts = {s.value: 0 for s in IssueSeverity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts


class AIQualityAssessor:
    """
    AI-powered topology quality assessment.
    
    Renders the mesh from multiple viewpoints and uses vision AI to
    identify topology issues and provide quality scores.
    """
    
    def __init__(
        self,
        api_provider: str = "openai",
        model: str = None,
        resolution: tuple[int, int] = (1024, 1024),
        num_views: int = 4,  # Fewer views than segmentation - just for assessment
        use_blender: bool = True,
        blender_path: str = None,
        wireframe_mode: bool = True,  # Show wireframe for topology analysis
    ):
        self.api_provider = api_provider
        self.model = model or self._default_model(api_provider)
        self.resolution = resolution
        self.num_views = num_views
        self.use_blender = use_blender
        self.blender_path = blender_path
        self.wireframe_mode = wireframe_mode
        self._renderer = None
    
    def _default_model(self, provider: str) -> str:
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
        }
        return defaults.get(provider, "gpt-4o")
    
    def _get_api_key(self) -> str:
        """Get API key from environment or keychain."""
        if self.api_provider == "openai":
            key = get_openai_api_key()
            if not key:
                raise ValueError("OpenAI API key not found")
            return key
        elif self.api_provider == "anthropic":
            key = get_anthropic_api_key()
            if not key:
                raise ValueError("Anthropic API key not found")
            return key
        else:
            raise ValueError(f"Unknown provider: {self.api_provider}")
    
    def assess(
        self,
        mesh: Mesh,
        original_mesh: Optional[Mesh] = None,
        context: Optional[str] = None,
    ) -> AIQualityReport:
        """
        Assess mesh topology quality using AI vision.
        
        Args:
            mesh: The retopologized mesh to assess
            original_mesh: Optional original mesh for comparison
            context: Optional context about the mesh (e.g., "character face for animation")
            
        Returns:
            AIQualityReport with scores, issues, and recommendations
        """
        logger.info(f"Starting AI quality assessment with {self.api_provider}/{self.model}")
        
        # Render the mesh
        renders = self._render_mesh(mesh)
        
        # Optionally render original for comparison
        original_renders = None
        if original_mesh is not None:
            original_renders = self._render_mesh(original_mesh, wireframe=False)
        
        # Build prompt and call API
        response = self._analyze_with_vision(renders, original_renders, context)
        
        # Parse response into report
        report = self._parse_response(response)
        
        logger.info(f"AI assessment complete: {report.overall_score:.1f}/100, "
                    f"{len(report.issues)} issues found")
        
        return report
    
    def _render_mesh(
        self,
        mesh: Mesh,
        wireframe: Optional[bool] = None,
    ) -> list[np.ndarray]:
        """Render mesh from multiple viewpoints."""
        if wireframe is None:
            wireframe = self.wireframe_mode
        
        if self.use_blender:
            return self._render_with_blender(mesh, wireframe)
        else:
            return self._render_fallback(mesh, wireframe)
    
    def _render_with_blender(
        self,
        mesh: Mesh,
        wireframe: bool,
    ) -> list[np.ndarray]:
        """Render using Blender subprocess."""
        try:
            import tempfile
            from neurotopo.analysis.blender_render import BlenderRenderer
            
            renderer = BlenderRenderer(
                blender_path=self.blender_path,
                resolution=self.resolution,
            )
            
            # Save mesh to temp file
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
                temp_path = f.name
            
            mesh.to_file(temp_path)
            
            # Use Blender renderer with wireframe option
            renders, _ = renderer.render_mesh(temp_path, wireframe=wireframe)
            
            # Clean up temp file
            import os
            os.unlink(temp_path)
            
            # Only return the number of views we need
            return renders[:self.num_views]
            
        except Exception as e:
            logger.warning(f"Blender render failed: {e}, using fallback")
            return self._render_fallback(mesh, wireframe)
    
    def _render_fallback(
        self,
        mesh: Mesh,
        wireframe: bool,
    ) -> list[np.ndarray]:
        """Simple fallback rendering."""
        # Use the semantic analyzer's fallback renderer
        from neurotopo.analysis.semantic import SemanticAnalyzer
        
        analyzer = SemanticAnalyzer(
            resolution=self.resolution,
            num_views=self.num_views,
            use_blender=False,
        )
        renders, _ = analyzer._render_views(mesh)
        return renders
    
    def _analyze_with_vision(
        self,
        renders: list[np.ndarray],
        original_renders: Optional[list[np.ndarray]],
        context: Optional[str],
    ) -> str:
        """Send renders to AI for analysis."""
        prompt = self._build_assessment_prompt(context)
        
        # Convert renders to base64
        images_b64 = [self._image_to_base64(r) for r in renders]
        
        if original_renders:
            original_b64 = [self._image_to_base64(r) for r in original_renders]
        else:
            original_b64 = None
        
        # Call appropriate API
        if self.api_provider == "openai":
            return self._call_openai(images_b64, original_b64, prompt)
        else:
            return self._call_anthropic(images_b64, original_b64, prompt)
    
    def _build_assessment_prompt(self, context: Optional[str]) -> str:
        """Build the assessment prompt."""
        base_prompt = """You are a technical 3D modeling expert evaluating polygon mesh topology for computer graphics applications.

These images show wireframe renders of a 3D polygon mesh that has been through an automatic retopology process. The blue lines represent polygon edges.

Your task is to analyze the mesh topology quality for use in animation, games, or rendering. This is purely a technical analysis of geometric structure.

Please evaluate:

1. **Edge Flow**: Do edges follow logical directions for the surface? Are there continuous edge loops where needed?

2. **Pole Placement**: Are poles (vertices with 3 or 5+ edges) in appropriate locations?

3. **Quad Quality**: Are quadrilateral faces roughly square? Any stretched or skewed quads?

4. **Density Distribution**: Is polygon density appropriate for the mesh complexity?

5. **Overall Topology**: Is this topology suitable for deformation/animation?

Return your assessment as JSON with this exact structure:
{
    "overall_score": 75,
    "mesh_type_detected": "3D model",
    "animation_ready": true,
    "summary": "Brief overall assessment of mesh topology quality",
    "strengths": ["Good edge flow", "Clean quad distribution"],
    "issues": [
        {
            "category": "pole_placement",
            "severity": "warning",
            "description": "Pole visible in high-deformation area",
            "location_hint": "center region",
            "recommendation": "Move pole to flatter area"
        }
    ],
    "recommended_actions": ["Add edge loops in key areas", "Improve quad regularity"]
}

Valid categories: edge_flow, pole_placement, quad_quality, density, deformation, silhouette, symmetry, general
Valid severities: info, warning, error, critical

Be specific and technical in your feedback."""

        if context:
            base_prompt += f"\n\nContext: {context}"
        
        base_prompt += "\n\nIMPORTANT: Respond with ONLY valid JSON. Do not add any explanatory text before or after the JSON object."
        
        return base_prompt
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64."""
        from PIL import Image
        
        img = Image.fromarray(image.astype(np.uint8))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _call_openai(
        self,
        images_b64: list[str],
        original_b64: Optional[list[str]],
        prompt: str,
    ) -> str:
        """Call OpenAI Vision API."""
        import openai
        
        api_key = self._get_api_key()
        client = openai.OpenAI(api_key=api_key)
        
        # Build message content
        content = [{"type": "text", "text": prompt}]
        
        # Add retopo mesh renders
        content.append({"type": "text", "text": "\n\nRetopologized mesh (wireframe views):"})
        for i, img_b64 in enumerate(images_b64):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"
                }
            })
        
        # Add original renders if provided
        if original_b64:
            content.append({"type": "text", "text": "\n\nOriginal mesh (for reference):"})
            for img_b64 in original_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                        "detail": "low"
                    }
                })
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=2000,
        )
        
        result = response.choices[0].message.content
        logger.debug(f"OpenAI response content: {result[:200] if result else 'None'}...")
        return result
    
    def _call_anthropic(
        self,
        images_b64: list[str],
        original_b64: Optional[list[str]],
        prompt: str,
    ) -> str:
        """Call Anthropic Vision API."""
        import anthropic
        
        api_key = self._get_api_key()
        client = anthropic.Anthropic(api_key=api_key)
        
        # Build content with images
        content = []
        
        content.append({"type": "text", "text": prompt})
        content.append({"type": "text", "text": "\n\nRetopologized mesh (wireframe views):"})
        
        for img_b64 in images_b64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                }
            })
        
        if original_b64:
            content.append({"type": "text", "text": "\n\nOriginal mesh (for reference):"})
            for img_b64 in original_b64:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    }
                })
        
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        
        return response.content[0].text
    
    def _parse_response(self, response: str) -> AIQualityReport:
        """Parse AI response into structured report."""
        # Handle None response
        if response is None:
            logger.warning("AI returned None response")
            return AIQualityReport(
                overall_score=50,
                summary="AI returned no response",
            )
        
        try:
            # Clean up response (remove markdown if present)
            text = response.strip()
            
            # Remove markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                # Find first code block
                parts = text.split("```")
                if len(parts) >= 3:
                    text = parts[1]
                    # Remove language identifier if present
                    if text.startswith(("json", "JSON")):
                        text = text[4:]
            
            text = text.strip()
            
            # Try to find JSON object in text
            if not text.startswith("{"):
                # Look for first { and last }
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    text = text[start:end+1]
            
            data = json.loads(text)
            
            # Parse issues
            issues = []
            for issue_data in data.get("issues", []):
                try:
                    issues.append(TopologyIssue(
                        category=IssueCategory(issue_data.get("category", "general")),
                        severity=IssueSeverity(issue_data.get("severity", "info")),
                        description=issue_data.get("description", ""),
                        location_hint=issue_data.get("location_hint"),
                        bbox_2d=issue_data.get("bbox_2d"),
                        view_index=issue_data.get("view_index"),
                        recommendation=issue_data.get("recommendation"),
                    ))
                except (ValueError, KeyError) as e:
                    logger.debug(f"Failed to parse issue: {e}")
            
            return AIQualityReport(
                overall_score=float(data.get("overall_score", 50)),
                issues=issues,
                strengths=data.get("strengths", []),
                summary=data.get("summary", ""),
                mesh_type_detected=data.get("mesh_type_detected"),
                animation_ready=data.get("animation_ready", False),
                recommended_actions=data.get("recommended_actions", []),
            )
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse AI response: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            # Return a minimal report with the summary from raw response
            return AIQualityReport(
                overall_score=50,
                summary=f"AI response (parse failed): {response[:500]}...",
            )


def assess_mesh_quality(
    mesh: Mesh,
    original_mesh: Optional[Mesh] = None,
    api_provider: str = "openai",
    context: Optional[str] = None,
) -> AIQualityReport:
    """
    Convenience function to assess mesh quality.
    
    Args:
        mesh: Retopologized mesh to assess
        original_mesh: Optional original mesh for comparison
        api_provider: "openai" or "anthropic"
        context: Optional context string
        
    Returns:
        AIQualityReport with assessment results
    """
    assessor = AIQualityAssessor(api_provider=api_provider)
    return assessor.assess(mesh, original_mesh, context)
